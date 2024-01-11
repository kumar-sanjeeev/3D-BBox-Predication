import torch


def get_corners(pred_offsets, confidence_scores):
    # Get the indices of the highest confidence score in each batch
    max_confidence_indices = torch.argmax(confidence_scores, dim=1)
    print(f"max_confidence_indices: {max_confidence_indices.shape}")
    print(f"max_confidence_indices: {max_confidence_indices}")

    output = torch.zeros((pred_offsets.shape[0], 8, 3))
    for b in range(0, pred_offsets.shape[0]):
        output[b, :, :] = pred_offsets[b, max_confidence_indices[b], :, :]

    return output, max_confidence_indices


# Example usage
batch_size = 4
num_points = 10

# Mock data for pred_offsets and confidence_scores
pred_offsets = torch.randn((batch_size, num_points, 8, 3))
confidence_scores = torch.rand((batch_size, num_points, 1))

# Get corners based on the highest confidence score
corners, idx = get_corners(pred_offsets, confidence_scores)

print(
    "Input shape - pred_offsets:",
    pred_offsets.shape,
    "confidence_scores:",
    confidence_scores.shape,
)
print("Output shape - corners:", corners.shape)
print("Output:", corners[0])
print(pred_offsets[0, idx[0], :, :])
