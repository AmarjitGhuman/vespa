# Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
vespa_add_library(searchlib_util OBJECT
    SOURCES
    bufferwriter.cpp
    comprbuffer.cpp
    comprfile.cpp
    dirtraverse.cpp
    drainingbufferwriter.cpp
    filealign.cpp
    fileheadertk.cpp
    filekit.cpp
    filesizecalculator.cpp
    fileutil.cpp
    foldedstringcompare.cpp
    logutil.cpp
    rawbuf.cpp
    slime_output_raw_buf_adapter.cpp
    state_explorer_utils.cpp
    stringenum.cpp
    url.cpp
    DEPENDS
)
