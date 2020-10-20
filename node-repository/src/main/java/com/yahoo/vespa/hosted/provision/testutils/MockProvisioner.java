// Copyright 2018 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.vespa.hosted.provision.testutils;

import com.yahoo.config.provision.ApplicationId;
import com.yahoo.config.provision.Capacity;
import com.yahoo.config.provision.ClusterSpec;
import com.yahoo.config.provision.HostFilter;
import com.yahoo.config.provision.HostSpec;
import com.yahoo.config.provision.ProvisionLock;
import com.yahoo.config.provision.ProvisionLogger;
import com.yahoo.config.provision.Provisioner;
import com.yahoo.transaction.NestedTransaction;

import java.util.Collection;
import java.util.List;

/**
 * @author freva
 */
public class MockProvisioner implements Provisioner {

    @Override
    public List<HostSpec> prepare(ApplicationId applicationId, ClusterSpec cluster, Capacity capacity, ProvisionLogger logger) {
        return List.of();
    }

    @Override
    public void activate(NestedTransaction transaction, Collection<HostSpec> hosts, ProvisionLock lock) {

    }

    @Override
    public void remove(NestedTransaction transaction, ProvisionLock lock) {

    }

    @Override
    public void restart(ApplicationId application, HostFilter filter) {

    }

    @Override
    public ProvisionLock lock(ApplicationId application) {
        return null;
    }

}
