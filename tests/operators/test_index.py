# """Test indexing operators"""

# import numpy as np
# import pytest

# from pygrog.operators.indexing import multi_index, multi_grid

# @pytest.mark.parametrize("dtype", [np.float32, np.complex64])
# @pytest.mark.parametrize("shape", [(4,), (3, 3)])
# def test_multi_index_basic(dtype, shape):
#     """Test that multi_index extracts correct values for simple cases."""
#     input_data = np.arange(np.prod(shape)).reshape(shape).astype(dtype)
#     indexes = np.array([0, 1, np.prod(shape) - 1])
#     expected = input_data.reshape(-1)[indexes]
#     result = multi_index(input_data, indexes, shape)

#     np.testing.assert_allclose(result, expected)
#     assert result.shape == (indexes.size,)


# @pytest.mark.parametrize("stack_shape", [None, (2,)])
# def test_multi_index_with_stack(stack_shape):
#     """Test multi_index with stacking dimensions."""
#     shape = (3,)
#     stacks = (2,) if stack_shape is not None else ()
#     input_data = np.arange(np.prod(shape) * (np.prod(stacks) or 1)).reshape(*stacks, *shape)
#     indexes = np.array([[0, 2], [1, 2]]) if stack_shape else np.array([0, 1])
#     result = multi_index(input_data, indexes, shape, stack_shape)

#     # Check that the output shape matches expected (..., *stacks, nsamples)
#     expected_shape = (*stacks, indexes.shape[-1])
#     assert result.shape == expected_shape


# @pytest.mark.parametrize("dtype", [np.float32, np.complex64])
# @pytest.mark.parametrize("shape", [(4,), (3, 2)])
# def test_multi_grid_inverse_of_multi_index(dtype, shape):
#     """Test that multi_grid is approximately inverse (adjoint) of multi_index."""
#     input_data = np.arange(np.prod(shape)).reshape(shape).astype(dtype)
#     indexes = np.arange(np.prod(shape))

#     extracted = multi_index(input_data, indexes, shape)
#     regridded = multi_grid(extracted, indexes, shape)

#     np.testing.assert_allclose(regridded, input_data)


# def test_multi_grid_with_stacks():
#     """Test gridding with multiple stack dimensions."""
#     shape = (3,)
#     stack_shape = (2,)
#     indexes = np.array([[0, 1, 2], [2, 1, 0]])
#     input_data = np.array([[10, 11, 12], [20, 21, 22]])

#     result = multi_grid(input_data, indexes, shape, stack_shape)
#     assert result.shape == (*stack_shape, *shape)
#     # Verify positions
#     np.testing.assert_allclose(result[0], [10, 11, 12])
#     np.testing.assert_allclose(result[1], [22, 21, 20])


# def test_multi_grid_duplicate_entries_accumulation():
#     """Test accumulation behavior when duplicate_entries=True."""
#     shape = (4,)
#     indexes = np.array([0, 1, 1, 3])
#     input_data = np.array([1.0, 2.0, 3.0, 4.0])

#     result = multi_grid(input_data, indexes, shape, duplicate_entries=True)
#     # index 1 should have accumulated 2+3 = 5
#     expected = np.array([1, 5, 0, 4])
#     np.testing.assert_allclose(result, expected)


# def test_shapes_consistency():
#     """Ensure consistent shapes for all combinations."""
#     input_data = np.random.randn(2, 3, 4)
#     indexes = np.array([0, 5, 7])
#     shape = (4,)
#     result = multi_index(input_data, indexes, shape)
#     assert result.shape[-1] == indexes.shape[-1]


# def test_invalid_index_raises():
#     """Invalid indices should raise an IndexError."""
#     shape = (4,)
#     input_data = np.arange(4)
#     indexes = np.array([0, 10])  # invalid
#     with pytest.raises(IndexError):
#         _ = multi_index(input_data, indexes, shape)