// Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#pragma once

#include "tensor.h"
#include <vespa/eval/eval/value.h>
#include <vespa/eval/eval/aggr.h>

namespace vespalib::tensor {

/**
 * A thin wrapper around a SimpleValue to be used as fallback for tensors with data
 * layouts not supported by the default tensor implementation.
 *
 * Tensor implementation class is currently inferred from its value
 * type. Consider adding explicit tagging to the tensor::Tensor
 * default implementation top-level class in the future.
 **/
class WrappedSimpleValue : public Tensor
{
private:
    std::unique_ptr<eval::Value> _space;
    const eval::Value &_tensor;
public:
    explicit WrappedSimpleValue(const eval::Value &tensor)
        : _space(), _tensor(tensor) {}
    explicit WrappedSimpleValue(std::unique_ptr<eval::Value> tensor)
        : _space(std::move(tensor)), _tensor(*_space) {}
    ~WrappedSimpleValue() {}
    const eval::Value &unwrap() const { return _tensor; }

    // Value API
    const eval::ValueType &type() const override { return _tensor.type(); }
    eval::TypedCells cells() const override { return _tensor.cells(); }
    const Index &index() const override { return _tensor.index(); }
    double as_double() const override { return _tensor.as_double(); }

    // tensor API
    eval::TensorSpec toSpec() const override;
    void accept(TensorVisitor &visitor) const override;
    MemoryUsage get_memory_usage() const override;

    Tensor::UP join(join_fun_t, const Tensor &) const override;
    Tensor::UP merge(join_fun_t, const Tensor &) const override;
    Tensor::UP reduce(join_fun_t, const std::vector<vespalib::string> &) const override;
    Tensor::UP reduce(eval::Aggr aggr, const std::vector<vespalib::string> &) const;

    Tensor::UP apply(const CellFunction &) const override;
    Tensor::UP modify(join_fun_t, const CellValues &) const override;
    Tensor::UP add(const Tensor &arg) const override;
    Tensor::UP remove(const CellValues &) const override;

    // extra functionality
    Tensor::UP concat(const Value &b, const vespalib::string &dimension);
    Tensor::UP rename(const std::vector<vespalib::string> &from, const std::vector<vespalib::string> &to);
};

} // namespace vespalib::tensor
