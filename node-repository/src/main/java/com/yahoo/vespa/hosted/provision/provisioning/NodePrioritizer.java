// Copyright 2018 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.vespa.hosted.provision.provisioning;

import com.yahoo.config.provision.ApplicationId;
import com.yahoo.config.provision.ClusterSpec;
import com.yahoo.config.provision.NodeResources;
import com.yahoo.config.provision.NodeType;
import com.yahoo.vespa.hosted.provision.LockedNodeList;
import com.yahoo.vespa.hosted.provision.Node;
import com.yahoo.vespa.hosted.provision.NodeList;
import com.yahoo.vespa.hosted.provision.NodeRepository;

import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Builds up data structures necessary for node prioritization. It wraps each node
 * up in a {@link NodeCandidate} object with attributes used in sorting.
 *
 * The prioritization logic is implemented by {@link NodeCandidate}.
 *
 * @author smorgrav
 */
public class NodePrioritizer {

    private final List<NodeCandidate> nodes = new ArrayList<>();
    private final LockedNodeList allNodes;
    private final HostCapacity capacity;
    private final NodeSpec requestedNodes;
    private final ApplicationId application;
    private final ClusterSpec clusterSpec;
    private final NodeRepository nodeRepository;
    private final boolean isDocker;
    private final boolean isAllocatingForReplacement;
    private final boolean isTopologyChange;
    private final int currentClusterSize;
    private final Set<Node> spareHosts;

    NodePrioritizer(LockedNodeList allNodes, ApplicationId application, ClusterSpec clusterSpec, NodeSpec nodeSpec,
                    int wantedGroups, NodeRepository nodeRepository) {
        this.allNodes = allNodes;
        this.capacity = new HostCapacity(allNodes, nodeRepository.resourcesCalculator());
        this.requestedNodes = nodeSpec;
        this.clusterSpec = clusterSpec;
        this.application = application;
        this.spareHosts = capacity.findSpareHosts(allNodes.asList(), nodeRepository.spareCount());
        this.nodeRepository = nodeRepository;

        NodeList nodesInCluster = allNodes.owner(application).type(clusterSpec.type()).cluster(clusterSpec.id());
        NodeList nonRetiredNodesInCluster = nodesInCluster.not().retired();
        long currentGroups = nonRetiredNodesInCluster.state(Node.State.active).stream()
                .flatMap(node -> node.allocation()
                        .flatMap(alloc -> alloc.membership().cluster().group().map(ClusterSpec.Group::index))
                        .stream())
                .distinct()
                .count();
        this.isTopologyChange = currentGroups != wantedGroups;

        this.currentClusterSize = (int) nonRetiredNodesInCluster.state(Node.State.active).stream()
                .map(node -> node.allocation().flatMap(alloc -> alloc.membership().cluster().group()))
                .filter(clusterSpec.group()::equals)
                .count();

        this.isAllocatingForReplacement = isReplacement(nodesInCluster.size(),
                                                        nodesInCluster.state(Node.State.failed).size());
        this.isDocker = resources(requestedNodes) != null;
    }

    /** Returns the list of nodes sorted by {@link NodeCandidate#compareTo(NodeCandidate)} */
    List<NodeCandidate> prioritize() {
        return nodes.stream().sorted().collect(Collectors.toList());
    }

    /**
     * Add nodes that have been previously reserved to the same application from
     * an earlier downsizing of a cluster
     */
    void addSurplusNodes(List<Node> surplusNodes) {
        for (Node node : surplusNodes) {
            NodeCandidate candidate = candidateFrom(node, true);
            if (!candidate.violatesSpares || isAllocatingForReplacement) {
                nodes.add(candidate);
            }
        }
    }

    /** Add a node on each docker host with enough capacity for the requested flavor  */
    void addNewDockerNodes() {
        if ( ! isDocker) return;

        LockedNodeList candidates = allNodes
                .filter(node -> node.type() != NodeType.host || nodeRepository.canAllocateTenantNodeTo(node))
                .filter(node -> node.reservedTo().isEmpty() || node.reservedTo().get().equals(application.tenant()));

        addNewDockerNodesOn(candidates);
    }

    private void addNewDockerNodesOn(LockedNodeList candidateHosts) {
        for (Node host : candidateHosts) {
            if ( spareHosts.contains(host) && !isAllocatingForReplacement) continue;
            if ( ! capacity.hasCapacity(host, resources(requestedNodes))) continue;
            if ( ! allNodes.childrenOf(host).owner(application).cluster(clusterSpec.id()).isEmpty()) continue;
            nodes.add(NodeCandidate.createNewChild(resources(requestedNodes),
                                                   capacity.freeCapacityOf(host, false),
                                                   host,
                                                   spareHosts.contains(host),
                                                   allNodes,
                                                   nodeRepository));
        }
    }

    /** Add existing nodes allocated to the application */
    void addApplicationNodes() {
        EnumSet<Node.State> legalStates = EnumSet.of(Node.State.active, Node.State.inactive, Node.State.reserved);
        allNodes.asList().stream()
                .filter(node -> node.type() == requestedNodes.type())
                .filter(node -> legalStates.contains(node.state()))
                .filter(node -> node.allocation().isPresent())
                .filter(node -> node.allocation().get().owner().equals(application))
                .filter(node -> node.state() == Node.State.active || canStillAllocateToParentOf(node))
                .map(node -> candidateFrom(node, false))
                .forEach(candidate -> nodes.add(candidate));
    }

    /** Add nodes already provisioned, but not allocated to any application */
    void addReadyNodes() {
        allNodes.asList().stream()
                .filter(node -> node.type() == requestedNodes.type())
                .filter(node -> node.state() == Node.State.ready)
                .map(node -> candidateFrom(node, false))
                .filter(n -> !n.violatesSpares || isAllocatingForReplacement)
                .forEach(candidate -> nodes.add(candidate));
    }

    /** Create a candidate from given pre-existing node */
    private NodeCandidate candidateFrom(Node node, boolean isSurplus) {
        Optional<Node> parent = allNodes.parentOf(node);
        if (parent.isPresent()) {
            return NodeCandidate.createChild(node,
                                             capacity.freeCapacityOf(parent.get(), false),
                                             parent.get(),
                                             spareHosts.contains(parent.get()),
                                             isSurplus,
                                             false,
                                             requestedNodes.canResize(node.resources(),
                                                                         capacity.freeCapacityOf(parent.get(), false),
                                                                         isTopologyChange,
                                                                         currentClusterSize));
        }
        else {
            return NodeCandidate.createStandalone(node, isSurplus, false);
        }
    }

    private boolean isReplacement(int nofNodesInCluster, int nodeFailedNodes) {
        if (nodeFailedNodes == 0) return false;

        return requestedNodes.fulfilledBy(nofNodesInCluster - nodeFailedNodes);
    }

    /**
     * We may regret that a non-active node is allocated to a host and not offer it to the application
     * now, e.g if we want to retire the host.
     *
     * @return true if we still want to allocate the given node to its parent
     */
    private boolean canStillAllocateToParentOf(Node node) {
        if (node.parentHostname().isEmpty()) return true;
        Optional<Node> parent = node.parentHostname().flatMap(nodeRepository::getNode);
        if (parent.isEmpty()) return false;
        return nodeRepository.canAllocateTenantNodeTo(parent.get());
    }

    private static NodeResources resources(NodeSpec requestedNodes) {
        if ( ! (requestedNodes instanceof NodeSpec.CountNodeSpec)) return null;
        return requestedNodes.resources().get();
    }

}
