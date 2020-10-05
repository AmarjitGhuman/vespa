// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#pragma once

#include <vespa/vespalib/util/arrayref.h>
#include <vespa/vespalib/stllike/string.h>
#include <vespa/vespalib/stllike/hash_map.h>
#include <vector>
#include <xxhash.h>
#include <type_traits>

namespace vespalib::eval {

/**
 * A wrapper around vespalib::hash_map, using it to map a list of
 * labels (a sparse address) to an integer value (dense subspace
 * index). Labels are stored in a separate vector to avoid
 * fragmentation caused by hash keys being vectors of values. Labels
 * can be specified in different ways during lookup and insert in
 * order to reduce the need for data restructuring when used to
 * integrate with the Value api. All labels are stored with a 64-bit
 * hash. This hash is used as label equality (assuming no
 * collisions). A order-sensitive 64bit hash constructed from
 * individual label hashes is used for address equality (also assuming
 * no collisions). The hash algorithm currently used is XXH3.
 *
 * 'add_mapping' will will bind the given address to an integer value
 * equal to the current (pre-insert) size of the map. The given
 * address MUST NOT already be in the map.
 *
 * 'lookup' will return the integer value associated with the
 * given address or a special npos value if the value is not found.
 **/
class SimpleSparseMap
{
public:
    using hash_t = XXH64_hash_t;

    static hash_t hash_label(const vespalib::string &str) {
        return XXH3_64bits(str.data(), str.size());
    }
    static hash_t hash_label(vespalib::stringref str) {
        return XXH3_64bits(str.data(), str.size());
    }
    static hash_t hash_label(const vespalib::stringref *str) {
        return XXH3_64bits(str->data(), str->size());
    }

    struct HashedLabel {
        vespalib::string label;
        hash_t hash;
        HashedLabel() : label(), hash(0) {}
        HashedLabel(const HashedLabel &rhs) = default;
        HashedLabel &operator=(const HashedLabel &rhs) = default;
        HashedLabel(HashedLabel &&rhs) = default;
        HashedLabel &operator=(HashedLabel &&rhs) = default;
        HashedLabel(const vespalib::string &str) : label(str), hash(hash_label(str)) {}
        HashedLabel(vespalib::stringref str) : label(str), hash(hash_label(str)) {}
        HashedLabel(const vespalib::stringref *str) : label(*str), hash(hash_label(*str)) {}
    };

    static hash_t hash_label(const HashedLabel &label) {
        return label.hash;
    }

    struct Key {
        uint32_t start;
        hash_t hash;
        Key() : start(0), hash(0) {}
        Key(uint32_t start_in, hash_t hash_in)
            : start(start_in), hash(hash_in) {}
    };

    struct Hash {
        hash_t operator()(const Key &key) const { return key.hash; }
        hash_t operator()(hash_t hash) const { return hash; }
    };

    struct Equal {
        bool operator()(const Key &a, hash_t b) const { return (a.hash == b); }
        bool operator()(const Key &a, const Key &b) const { return (a.hash == b.hash); }
    };

    using MapType = vespalib::hash_map<Key,uint32_t,Hash,Equal>;

private:
    size_t _num_dims;
    std::vector<HashedLabel> _labels;
    MapType _map;

public:
    SimpleSparseMap(size_t num_dims_in, size_t expected_subspaces)
        : _num_dims(num_dims_in), _labels(), _map(expected_subspaces * 2)
    {
        _labels.reserve(_num_dims * expected_subspaces);
    }
    ~SimpleSparseMap();
    size_t size() const { return _map.size(); }
    size_t num_dims() const { return _num_dims; }
    static constexpr size_t npos() { return -1; }
    const std::vector<HashedLabel> &labels() const { return _labels; }

    ConstArrayRef<HashedLabel> make_addr(uint32_t start) const {
        return ConstArrayRef<HashedLabel>(&_labels[start], _num_dims);
    }

    template <typename T>
    hash_t hash_addr(ConstArrayRef<T> addr) const {
        hash_t h = 0;
        for (const auto &label: addr) {
            h = 31 * h + hash_label(label);
        }
        return h;
    }

    template <typename T>
    void add_mapping(ConstArrayRef<T> addr, hash_t hash) {
        uint32_t value = _map.size();
        uint32_t start = _labels.size();
        for (const auto &label: addr) {
            _labels.emplace_back(label);
        }
        _map.insert(std::make_pair(Key(start, hash), value));
    }

    template <typename T>
    void add_mapping(ConstArrayRef<T> addr) {
        hash_t h = 0;
        uint32_t value = _map.size();
        uint32_t start = _labels.size();
        for (const auto &label: addr) {
            _labels.emplace_back(label);
            h = 31 * h + hash_label(_labels.back());
        }
        _map.insert(std::make_pair(Key(start, h), value));
    }

    size_t lookup(hash_t hash) const {
        auto pos = _map.find(hash);
        return (pos == _map.end()) ? npos() : pos->second;
    }

    template <typename T>
    size_t lookup(ConstArrayRef<T> addr) const {
        return lookup(hash_addr(addr));
    }

    template <typename F>
    void each_map_entry(F &&f) const {
        _map.for_each([&](const auto &entry)
                      {
                          f(entry.first.start, entry.second, entry.first.hash);
                      });
    }
};

}
