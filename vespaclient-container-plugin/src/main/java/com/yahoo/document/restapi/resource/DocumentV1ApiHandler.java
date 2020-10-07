// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.document.restapi.resource;

import com.fasterxml.jackson.core.JsonFactory;
import com.google.inject.Inject;
import com.yahoo.cloud.config.ClusterListConfig;
import com.yahoo.container.core.documentapi.VespaDocumentAccess;
import com.yahoo.document.DocumentId;
import com.yahoo.document.DocumentOperation;
import com.yahoo.document.DocumentPut;
import com.yahoo.document.DocumentTypeManager;
import com.yahoo.document.DocumentUpdate;
import com.yahoo.document.TestAndSetCondition;
import com.yahoo.document.config.DocumentmanagerConfig;
import com.yahoo.document.json.DocumentOperationType;
import com.yahoo.document.json.JsonReader;
import com.yahoo.document.json.JsonWriter;
import com.yahoo.document.restapi.DocumentOperationExecutor;
import com.yahoo.document.restapi.DocumentOperationExecutor.ErrorType;
import com.yahoo.document.restapi.DocumentOperationExecutor.Group;
import com.yahoo.document.restapi.DocumentOperationExecutor.OperationContext;
import com.yahoo.document.restapi.DocumentOperationExecutor.VisitOperationsContext;
import com.yahoo.document.restapi.DocumentOperationExecutor.VisitorOptions;
import com.yahoo.document.restapi.DocumentOperationExecutorConfig;
import com.yahoo.document.restapi.DocumentOperationExecutorImpl;
import com.yahoo.documentapi.DocumentOperationParameters;
import com.yahoo.documentapi.metrics.DocumentApiMetrics;
import com.yahoo.documentapi.metrics.DocumentOperationStatus;
import com.yahoo.jdisc.Metric;
import com.yahoo.jdisc.Request;
import com.yahoo.jdisc.Response;
import com.yahoo.jdisc.handler.AbstractRequestHandler;
import com.yahoo.jdisc.handler.CompletionHandler;
import com.yahoo.jdisc.handler.ContentChannel;
import com.yahoo.jdisc.handler.ReadableContentChannel;
import com.yahoo.jdisc.handler.ResponseHandler;
import com.yahoo.jdisc.handler.UnsafeContentInputStream;
import com.yahoo.container.core.HandlerMetricContextUtil;
import com.yahoo.jdisc.http.HttpRequest;
import com.yahoo.jdisc.http.HttpRequest.Method;
import com.yahoo.metrics.simple.MetricReceiver;
import com.yahoo.restapi.Path;
import com.yahoo.slime.Cursor;
import com.yahoo.slime.Inspector;
import com.yahoo.slime.Slime;
import com.yahoo.slime.SlimeUtils;
import com.yahoo.vespa.config.content.AllClustersBucketSpacesConfig;
import com.yahoo.yolean.Exceptions;

import java.io.InputStream;
import java.nio.ByteBuffer;
import java.time.Clock;
import java.time.Instant;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.logging.Logger;

import static com.yahoo.documentapi.DocumentOperationParameters.parameters;
import static com.yahoo.jdisc.http.HttpRequest.Method.DELETE;
import static com.yahoo.jdisc.http.HttpRequest.Method.GET;
import static com.yahoo.jdisc.http.HttpRequest.Method.OPTIONS;
import static com.yahoo.jdisc.http.HttpRequest.Method.POST;
import static com.yahoo.jdisc.http.HttpRequest.Method.PUT;
import static java.util.Objects.requireNonNull;
import static java.util.logging.Level.FINE;
import static java.util.logging.Level.SEVERE;
import static java.util.logging.Level.WARNING;
import static java.util.stream.Collectors.joining;

/**
 * Asynchronous HTTP handler for /document/v1/
 *
 * @author jonmv
 */
public class DocumentV1ApiHandler extends AbstractRequestHandler {

    private static final Logger log = Logger.getLogger(DocumentV1ApiHandler.class.getName());
    private static final Parser<Integer> numberParser = Integer::parseInt;
    private static final Parser<Boolean> booleanParser = Boolean::parseBoolean;

    private static final CompletionHandler logException = new CompletionHandler() {
        @Override public void completed() { }
        @Override public void failed(Throwable t) {
            log.log(FINE, () -> "Exception writing or closing response data: " + Exceptions.toMessageString(t));
        }
    };

    private static final ContentChannel ignoredContent = new ContentChannel() {
        @Override public void write(ByteBuffer buf, CompletionHandler handler) { handler.completed(); }
        @Override public void close(CompletionHandler handler) { handler.completed(); }
    };

    private static final String CREATE = "create";
    private static final String CONDITION = "condition";
    private static final String ROUTE = "route"; // TODO jonmv: set for everything except Get
    private static final String FIELD_SET = "fieldSet";
    private static final String SELECTION = "selection";
    private static final String CLUSTER = "cluster"; // TODO jonmv: set for Get
    private static final String CONTINUATION = "continuation";
    private static final String WANTED_DOCUMENT_COUNT = "wantedDocumentCount";
    private static final String CONCURRENCY = "concurrency";
    private static final String BUCKET_SPACE = "bucketSpace";

    private final Clock clock;
    private final Metric metric; // TODO jonmv: make response class which logs on completion/error
    private final DocumentApiMetrics metrics;
    private final DocumentOperationExecutor executor;
    private final DocumentOperationParser parser;
    private final Map<String, Map<Method, Handler>> handlers;
    private final AtomicReference<VisitOperationsContext> lastVisit = new AtomicReference<>();
    private final Queue<OperationContext> contexts = new ConcurrentLinkedQueue<>();

    @Inject
    public DocumentV1ApiHandler(Metric metric,
                                MetricReceiver metricReceiver,
                                VespaDocumentAccess documentAccess,
                                DocumentmanagerConfig documentManagerConfig,
                                ClusterListConfig clusterListConfig,
                                AllClustersBucketSpacesConfig bucketSpacesConfig,
                                DocumentOperationExecutorConfig executorConfig) {
        this(Clock.systemUTC(),
             new DocumentOperationExecutorImpl(clusterListConfig, bucketSpacesConfig, executorConfig, documentAccess, Clock.systemUTC()),
             new DocumentOperationParser(documentManagerConfig),
             metric,
             metricReceiver);
    }

    DocumentV1ApiHandler(Clock clock, DocumentOperationExecutor executor, DocumentOperationParser parser,
                         Metric metric, MetricReceiver metricReceiver) {
        this.clock = clock;
        this.executor = executor;
        this.parser = parser;
        this.metric = metric;
        this.metrics = new DocumentApiMetrics(metricReceiver, "documentV1");
        this.handlers = defineApi();
    }

    @Override
    public ContentChannel handleRequest(Request rawRequest, ResponseHandler rawResponseHandler) {
        try {
            HandlerMetricContextUtil.onHandle(rawRequest, metric, getClass());
            ResponseHandler responseHandler = response -> {
                try {
                    HandlerMetricContextUtil.onHandled(rawRequest, metric, getClass());
                    return rawResponseHandler.handleResponse(response);
                }
                catch (Throwable t) {
                    log.log(SEVERE, "Uncaught during metric reporting", t);
                    throw t;
                }
            };

            HttpRequest request = (HttpRequest) rawRequest;
            try {
                Path requestPath = new Path(request.getUri());
                for (String path : handlers.keySet())
                    if (requestPath.matches(path)) {
                        Map<Method, Handler> methods = handlers.get(path);
                        if (methods.containsKey(request.getMethod()))
                            return methods.get(request.getMethod()).handle(request, new DocumentPath(requestPath), responseHandler);

                        if (request.getMethod() == OPTIONS)
                            return options(methods.keySet(), responseHandler);

                        return methodNotAllowed(request, methods.keySet(), responseHandler);
                    }
                return notFound(request, handlers.keySet(), responseHandler);
            }
            catch (IllegalArgumentException e) {
                return badRequest(request, e, responseHandler);
            }
            catch (RuntimeException e) {
                return serverError(request, e, responseHandler);
            }
        }
        catch (Throwable t) {
            log.log(SEVERE, "Uncaught during handle", t);
            throw t;
        }
    }

    @Override
    public void destroy() {
        this.executor.shutdown();
    }

    private Map<String, Map<Method, Handler>> defineApi() {
        Map<String, Map<Method, Handler>> handlers = new LinkedHashMap<>();

        handlers.put("/document/v1/",
                     Map.of(GET, this::getRoot));

        handlers.put("/document/v1/{namespace}/{documentType}/docid/",
                     Map.of(GET, this::getDocumentType));

        handlers.put("/document/v1/{namespace}/{documentType}/group/{group}/",
                     Map.of(GET, this::getDocumentType));

        handlers.put("/document/v1/{namespace}/{documentType}/number/{number}/",
                     Map.of(GET, this::getDocumentType));

        handlers.put("/document/v1/{namespace}/{documentType}/docid/{docid}",
                     Map.of(GET, this::getDocument,
                            POST, this::postDocument,
                            PUT, this::putDocument,
                            DELETE, this::deleteDocument));

        handlers.put("/document/v1/{namespace}/{documentType}/group/{group}/{docid}",
                     Map.of(GET, this::getDocument,
                            POST, this::postDocument,
                            PUT, this::putDocument,
                            DELETE, this::deleteDocument));

        handlers.put("/document/v1/{namespace}/{documentType}/number/{number}/{docid}",
                     Map.of(GET, this::getDocument,
                            POST, this::postDocument,
                            PUT, this::putDocument,
                            DELETE, this::deleteDocument));

        return Collections.unmodifiableMap(handlers);
    }

    private ContentChannel getRoot(HttpRequest request, DocumentPath path, ResponseHandler handler) {
        Cursor root = responseRoot(request);
        VisitOperationsContext context = visitorContext(request, root, root.setArray("documents"), handler);
        lastVisit.set(context);
        executor.visit(parseOptions(request, path).build(), context);
        return ignoredContent;
    }

    private ContentChannel getDocumentType(HttpRequest request, DocumentPath path, ResponseHandler handler) {
        Cursor root = responseRoot(request);
        VisitorOptions.Builder options = parseOptions(request, path);
        options = options.documentType(path.documentType());
        options = options.namespace(path.namespace());
        options = path.group().map(options::group).orElse(options);
        VisitOperationsContext context = visitorContext(request, root, root.setArray("documents"), handler);
        lastVisit.set(context);
        executor.visit(options.build(), context);
        return ignoredContent;
    }

    private static VisitOperationsContext visitorContext(HttpRequest request, Cursor root, Cursor documents, ResponseHandler handler) {
        Object monitor = new Object();
        return new VisitOperationsContext((type, message) -> {
                                              synchronized (monitor) {
                                                  handleError(request, type, message, root, handler);
                                              }
                                          },
                                          token -> {
                                              token.ifPresent(value -> root.setString("continuation", value));
                                              synchronized (monitor) {
                                                  respond(root, handler);
                                              }
                                          },
                                          // TODO jonmv: make streaming — first doc indicates 200 OK anyway — unless session dies, which is a semi-200 anyway
                                          document -> {
                                              try {
                                                  synchronized (monitor) { // Putting things into the slime is not thread safe, so need synchronization.
                                                      SlimeUtils.copyObject(SlimeUtils.jsonToSlime(JsonWriter.toByteArray(document)).get(),
                                                                            documents.addObject());
                                                  }
                                              }
                                              // TODO jonmv: This shouldn't happen much, but ... expose errors too?
                                              catch (RuntimeException e) {
                                                  log.log(WARNING, "Exception serializing document in document/v1 visit response", e);
                                              }
                                          });
    }

    private ContentChannel getDocument(HttpRequest request, DocumentPath path, ResponseHandler handler) {
        DocumentId id = path.id();
        DocumentOperationParameters parameters = parameters();
        parameters = getProperty(request, CLUSTER).map(executor::routeToCluster).map(parameters::withRoute).orElse(parameters);
        parameters = getProperty(request, FIELD_SET).map(parameters::withFieldSet).orElse(parameters);
        OperationContext context =                      new OperationContext((type, message) -> handleError(request, type, message, responseRoot(request, id), handler),
                                                                             document -> {
                                                                                 try {
                                                                                     Cursor root = responseRoot(request, id);
                                                                                     document.map(JsonWriter::toByteArray)
                                                                                             .map(SlimeUtils::jsonToSlime)
                                                                                             .ifPresent(doc -> SlimeUtils.copyObject(doc.get().field("fields"), root.setObject("fields")));
                                                                                     respond(document.isPresent() ? 200 : 404,
                                                                                             root,
                                                                                             handler);
                                                                                 }
                                                                                 catch (Exception e) {
                                                                                     serverError(request, new RuntimeException(e), handler);
                                                                                 }
                                                                             });
        contexts.add(context);
        executor.get(id,
                     parameters,
                                          context);
        return ignoredContent;
    }

    private ContentChannel postDocument(HttpRequest request, DocumentPath path, ResponseHandler rawHandler) {
        DocumentId id = path.id();
        ResponseHandler handler = new MeasuringResponseHandler(rawHandler, com.yahoo.documentapi.metrics.DocumentOperationType.PUT, clock.instant());
        return new ForwardingContentChannel(in -> {
            try {
                DocumentPut put = parser.parsePut(in, id.toString());
                getProperty(request, CONDITION).map(TestAndSetCondition::new).ifPresent(put::setCondition);
                executor.put(put,
                             getProperty(request, ROUTE).map(parameters()::withRoute).orElse(parameters()),
                             new OperationContext((type, message) -> handleError(request, type, message, responseRoot(request, id), handler),
                                                  __ -> respond(responseRoot(request, id), handler)));
            }
            catch (IllegalArgumentException e) {
                badRequest(request, e, handler);
            }
            catch (RuntimeException e) {
                serverError(request, e, handler);
            }
        });
    }

    private ContentChannel putDocument(HttpRequest request, DocumentPath path, ResponseHandler rawHandler) {
        DocumentId id = path.id();
        ResponseHandler handler = new MeasuringResponseHandler(rawHandler, com.yahoo.documentapi.metrics.DocumentOperationType.UPDATE, clock.instant());
        return new ForwardingContentChannel(in -> {
            try {
                DocumentUpdate update = parser.parseUpdate(in, id.toString());
                getProperty(request, CONDITION).map(TestAndSetCondition::new).ifPresent(update::setCondition);
                getProperty(request, CREATE).map(booleanParser::parse).ifPresent(update::setCreateIfNonExistent);
                executor.update(update,
                                getProperty(request, ROUTE).map(parameters()::withRoute).orElse(parameters()),
                                new OperationContext((type, message) -> handleError(request, type, message, responseRoot(request, id), handler),
                                                     __ -> respond(responseRoot(request, id), handler)));
            }
            catch (IllegalArgumentException e) {
                badRequest(request, e, handler);
            }
            catch (RuntimeException e) {
                serverError(request, e, handler);
            }
        });
    }

    private ContentChannel deleteDocument(HttpRequest request, DocumentPath path, ResponseHandler rawHandler) {
        DocumentId id = path.id();
        ResponseHandler handler = new MeasuringResponseHandler(rawHandler, com.yahoo.documentapi.metrics.DocumentOperationType.REMOVE, clock.instant());
        executor.remove(id,
                        getProperty(request, ROUTE).map(parameters()::withRoute).orElse(parameters()),
                        new OperationContext((type, message) -> handleError(request, type, message, responseRoot(request, id), handler),
                                             __ -> respond(responseRoot(request, id), handler)));
        return ignoredContent;
    }

    private static void handleError(HttpRequest request, ErrorType type, String message, Cursor root, ResponseHandler handler) {
        switch (type) {
            case BAD_REQUEST:
                badRequest(request, message, root, handler);
                break;
            case NOT_FOUND:
                notFound(request, message, root, handler);
                break;
            case PRECONDITION_FAILED:
                preconditionFailed(request, message, root, handler);
                break;
            case OVERLOAD:
                overload(request, message, root, handler);
                break;
            case TIMEOUT:
                timeout(request, message, root, handler);
                break;
            case INSUFFICIENT_STORAGE:
                insufficientStorage(request, message, root, handler);
                break;
            default:
                log.log(WARNING, "Unexpected error type '" + type + "'");
            case ERROR: // intentional fallthrough
                serverError(request, message, root, handler);
        }
    }

    // ------------------------------------------------ Responses ------------------------------------------------

    private static Cursor responseRoot(HttpRequest request) {
        Cursor root = new Slime().setObject();
        root.setString("pathId", request.getUri().getRawPath());
        return root;
    }

    private static Cursor responseRoot(HttpRequest request, DocumentId id) {
        Cursor root = responseRoot(request);
        root.setString("id", id.toString());
        return root;
    }

    private static ContentChannel options(Collection<Method> methods, ResponseHandler handler) {
        Response response = new Response(Response.Status.NO_CONTENT);
        response.headers().add("Allow", methods.stream().sorted().map(Method::name).collect(joining(",")));
        handler.handleResponse(response).close(logException);
        return ignoredContent;
    }

    private static ContentChannel badRequest(HttpRequest request, IllegalArgumentException e, ResponseHandler handler) {
        return badRequest(request, Exceptions.toMessageString(e), responseRoot(request), handler);
    }

    private static ContentChannel badRequest(HttpRequest request, String message, Cursor root, ResponseHandler handler) {
        log.log(FINE, () -> "Bad request for " + request.getMethod() + " at " + request.getUri().getRawPath() + ": " + message);
        root.setString("message", message);
        return respond(Response.Status.BAD_REQUEST, root, handler);
    }

    private static ContentChannel notFound(HttpRequest request, Collection<String> paths, ResponseHandler handler) {
        return notFound(request,
                        "Nothing at '" + request.getUri().getRawPath() + "'. " +
                        "Available paths are:\n" + String.join("\n", paths),
                        responseRoot(request),
                        handler);
    }

    private static ContentChannel notFound(HttpRequest request, String message, Cursor root, ResponseHandler handler) {
        root.setString("message", message);
        return respond(Response.Status.NOT_FOUND, root, handler);
    }

    private static ContentChannel methodNotAllowed(HttpRequest request, Collection<Method> methods, ResponseHandler handler) {
        Cursor root = responseRoot(request);
        root.setString("message",
                       "'" + request.getMethod() + "' not allowed at '" + request.getUri().getRawPath() + "'. " +
                       "Allowed methods are: " + methods.stream().sorted().map(Method::name).collect(joining(", ")));
        return respond(Response.Status.METHOD_NOT_ALLOWED,
                       root,
                       handler);
    }

    private static ContentChannel preconditionFailed(HttpRequest request, String message, Cursor root, ResponseHandler handler) {
        root.setString("message", message);
        return respond(Response.Status.PRECONDITION_FAILED, root, handler);
    }

    private static ContentChannel overload(HttpRequest request, String message, Cursor root, ResponseHandler handler) {
        log.log(FINE, () -> "Overload handling request " + request.getMethod() + " " + request.getUri().getRawPath() + ": " + message);
        root.setString("message", message);
        return respond(Response.Status.TOO_MANY_REQUESTS, root, handler);
    }

    private static ContentChannel serverError(HttpRequest request, RuntimeException e, ResponseHandler handler) {
        log.log(WARNING, "Uncaught exception handling request " + request.getMethod() + " " + request.getUri().getRawPath() + ":", e);
        Cursor root = responseRoot(request);
        root.setString("message", Exceptions.toMessageString(e));
        return respond(Response.Status.INTERNAL_SERVER_ERROR, root, handler);
    }

    private static ContentChannel serverError(HttpRequest request, String message, Cursor root, ResponseHandler handler) {
        log.log(WARNING, "Uncaught exception handling request " + request.getMethod() + " " + request.getUri().getRawPath() + ": " + message);
        root.setString("message", message);
        return respond(Response.Status.INTERNAL_SERVER_ERROR, root, handler);
    }

    private static ContentChannel timeout(HttpRequest request, String message, Cursor root, ResponseHandler handler) {
        log.log(FINE, () -> "Timeout handling request " + request.getMethod() + " " + request.getUri().getRawPath() + ": " + message);
        root.setString("message", message);
        return respond(Response.Status.GATEWAY_TIMEOUT, root, handler);
    }

    private static ContentChannel insufficientStorage(HttpRequest request, String message, Cursor root, ResponseHandler handler) {
        log.log(FINE, () -> "Insufficient storage for " + request.getMethod() + " " + request.getUri().getRawPath() + ": " + message);
        root.setString("message", message);
        return respond(Response.Status.INSUFFICIENT_STORAGE, root, handler);
    }

    private static ContentChannel respond(Inspector root, ResponseHandler handler) {
        return respond(200, root, handler);
    }

    private static ContentChannel respond(int status, Inspector root, ResponseHandler handler) {
        Response response = new Response(status);
        response.headers().put("Content-Type", "application/json; charset=UTF-8");
        ContentChannel out = null;
        try {
            out = handler.handleResponse(response);
            out.write(ByteBuffer.wrap(Exceptions.uncheck(() -> SlimeUtils.toJsonBytes(root))), logException);
        }
        catch (Exception e) {
            log.log(FINE, () -> "Problems writing data to jDisc content channel: " + Exceptions.toMessageString(e));
        }
        finally {
            if (out != null) try {
                out.close(logException);
            }
            catch (Exception e) {
                log.log(FINE, () -> "Problems closing jDisc content channel: " + Exceptions.toMessageString(e));
            }
        }
        return ignoredContent;
    }

    // ------------------------------------------------ Helpers ------------------------------------------------

    private VisitorOptions.Builder parseOptions(HttpRequest request, DocumentPath path) {
        VisitorOptions.Builder options = VisitorOptions.builder();

        getProperty(request, SELECTION).ifPresent(options::selection);
        getProperty(request, CONTINUATION).ifPresent(options::continuation);
        getProperty(request, FIELD_SET).ifPresent(options::fieldSet);
        getProperty(request, CLUSTER).ifPresent(options::cluster);
        getProperty(request, BUCKET_SPACE).ifPresent(options::bucketSpace);
        getProperty(request, WANTED_DOCUMENT_COUNT, numberParser)
                .ifPresent(count -> options.wantedDocumentCount(Math.min(1 << 10, count)));
        getProperty(request, CONCURRENCY, numberParser)
                .ifPresent(concurrency -> options.concurrency(Math.min(100, concurrency)));

        return options;
    }

    static class DocumentPath {

        private final Path path;
        private final Optional<Group> group;

        DocumentPath(Path path) {
            this.path = requireNonNull(path);
            this.group = Optional.ofNullable(path.get("number")).map(numberParser::parse).map(Group::of)
                                 .or(() -> Optional.ofNullable(path.get("group")).map(Group::of));
        }

        DocumentId id() {
            return new DocumentId("id:" + requireNonNull(path.get("namespace")) +
                                  ":" + requireNonNull(path.get("documentType")) +
                                  ":" + group.map(Group::docIdPart).orElse("") +
                                  ":" + requireNonNull(path.get("docid")));
        }

        String documentType() { return requireNonNull(path.get("documentType")); }
        String namespace() { return requireNonNull(path.get("namespace")); }
        Optional<Group> group() { return group; }

    }

    private static Optional<String> getProperty(HttpRequest request, String name) {
        List<String> values = request.parameters().get(name);
        if (values != null && values.size() != 0)
            return Optional.ofNullable(values.get(values.size() - 1));

        return Optional.empty();
    }

    private static <T> Optional<T> getProperty(HttpRequest request, String name, Parser<T> parser) {
        return getProperty(request, name).map(parser::parse);
    }


    @FunctionalInterface
    interface Parser<T> extends Function<String, T> {
        default T parse(String value) {
            try {
                return apply(value);
            }
            catch (RuntimeException e) {
                throw new IllegalArgumentException("Failed parsing '" + value + "': " + Exceptions.toMessageString(e));
            }
        }
    }


    @FunctionalInterface
    interface Handler {
        ContentChannel handle(HttpRequest request, DocumentPath path, ResponseHandler handler);
    }


    /** Readable content channel which forwards data to a reader when closed. */
    static class ForwardingContentChannel implements ContentChannel {

        private final ReadableContentChannel delegate = new ReadableContentChannel();
        private final Consumer<InputStream> reader;

        public ForwardingContentChannel(Consumer<InputStream> reader) {
            this.reader = reader;
        }

        /** Write is complete when we have stored the buffer — call completion handler. */
        @Override
        public void write(ByteBuffer buf, CompletionHandler handler) {
            try {
                delegate.write(buf, logException);
                handler.completed();
            }
            catch (Exception e) {
                handler.failed(e);
            }
        }

        /** Close is complete when we have close the buffer. */
        @Override
        public void close(CompletionHandler handler) {
            try {
                delegate.close(logException);
                try (UnsafeContentInputStream in = new UnsafeContentInputStream(delegate)) {
                    reader.accept(in);
                }
                handler.completed();
            }
            catch (Exception e) {
                handler.failed(e);
            }
        }

    }


    static class DocumentOperationParser {

        private static final JsonFactory jsonFactory = new JsonFactory();

        private final DocumentTypeManager manager;

        DocumentOperationParser(DocumentmanagerConfig config) {
            this.manager = new DocumentTypeManager(config);
        }

        DocumentPut parsePut(InputStream inputStream, String docId) {
            return (DocumentPut) parse(inputStream, docId, DocumentOperationType.PUT);
        }

        DocumentUpdate parseUpdate(InputStream inputStream, String docId)  {
            return (DocumentUpdate) parse(inputStream, docId, DocumentOperationType.UPDATE);
        }

        private DocumentOperation parse(InputStream inputStream, String docId, DocumentOperationType operation)  {
            return new JsonReader(manager, inputStream, jsonFactory).readSingleDocument(operation, docId);
        }

    }

    private class MeasuringResponseHandler implements ResponseHandler {

        private final ResponseHandler delegate;
        private final com.yahoo.documentapi.metrics.DocumentOperationType type;
        private final Instant start;

        private MeasuringResponseHandler(ResponseHandler delegate, com.yahoo.documentapi.metrics.DocumentOperationType type, Instant start) {
            this.delegate = delegate;
            this.type = type;
            this.start = start;
        }

        @Override
        public ContentChannel handleResponse(Response response) {
            switch (response.getStatus() / 100) {
                case 2: metrics.reportSuccessful(type, start); break;
                case 4: metrics.reportFailure(type, DocumentOperationStatus.REQUEST_ERROR); break;
                case 5: metrics.reportFailure(type, DocumentOperationStatus.SERVER_ERROR); break;
            }
            return delegate.handleResponse(response);
        }

    }

}
