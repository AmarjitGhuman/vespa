// Copyright 2019 Oath Inc. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#pragma once

#include <cstring>

namespace search::datastore {

/*
 * Class containing common metadata for entries in unique store.
 */
class UniqueStoreEntryBase {
    mutable uint32_t _ref_count;
protected:
    constexpr UniqueStoreEntryBase()
        : _ref_count(0u)
    {
    }
public:
    uint32_t get_ref_count() const { return _ref_count; }
    void set_ref_count(uint32_t ref_count) const { _ref_count = ref_count; }
    void inc_ref_count() const { ++_ref_count; }
    void dec_ref_count() const { --_ref_count; }
};

}