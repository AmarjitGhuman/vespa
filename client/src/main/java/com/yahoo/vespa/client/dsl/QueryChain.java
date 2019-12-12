// Copyright 2019 Oath Inc. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.vespa.client.dsl;

public abstract class QueryChain {

    String op;
    int index; // for distinct each query chain
    Sources sources;
    Select select;
    Query query;
    boolean nonEmpty;

    void setOp(String op) {
        this.op = op;
    }

    String getOp() {
        return op;
    }

    void setIndex(int index) {
        this.index = index;
    }

    Sources getSources() {
        return sources;
    }

    void setSources(Sources sources) {
        this.sources = sources;
    }

    Select getSelect() {
        return select;
    }

    Query getQuery() {
        return query;
    }

    boolean nonEmpty() {
        return nonEmpty;
    }

    abstract boolean hasPositiveSearchField(String fieldName);

    abstract boolean hasPositiveSearchField(String fieldName, Object value);

    abstract boolean hasNegativeSearchField(String fieldName);

    abstract boolean hasNegativeSearchField(String fieldName, Object value);
}
