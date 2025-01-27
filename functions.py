import torch
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
from skimage.metrics import structural_similarity as ssim
from torchvision.transforms import functional as TF
import random
from skimage.color import gray2rgb, rgb2gray, label2rgb
from lime.wrappers.scikit_image import SegmentationAlgorithm
from lime import lime_image
from functools import partial

# ==> CHOOSE THE BEST SEGMENTATION ALGORITHM <==
segmenter = SegmentationAlgorithm('slic', n_segments=100, compactness=10)
# segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=150, ratio=0.2)
# segmenter = SegmentationAlgorithm('felzenszwalb', scale=100, sigma=0.8, min_size=5)

# Initialize the LIME Image Explainer
explainer = lime_image.LimeImageExplainer()

## FUNCTIONS FOR INTEGRATED GRADIENTS

# Function to calculate completeness gap
def compute_completeness_gap(model, input_image, baseline, target_class):
    """
    Computes the completeness gap for the Integrated Gradients attributions.

    Args:
        model: The model to evaluate.
        input_image: Input image tensor of shape (1, C, H, W).
        baseline: Baseline image tensor of shape (1, C, H, W).
        target_class: The target class index for which the attributions are computed.

    Returns:
        completeness_gap: The absolute difference between the sum of attributions and the model's prediction difference.
    """
    # Initialize Integrated Gradients
    ig = IntegratedGradients(model)

    # Compute attributions
    attributions = ig.attribute(input_image, baseline, target=target_class)

    # Compute prediction for the input and baseline
    input_prediction = model(input_image)[:, target_class].item()
    baseline_prediction = model(baseline)[:, target_class].item()

    # Sum of attributions
    attribution_sum = attributions.sum().item()

    # Compute completeness gap
    completeness_gap = abs(attribution_sum - (input_prediction - baseline_prediction))

    return completeness_gap, attributions



# Function to compute Sensitivity Analysis
def compute_sensitivity_analysis(model, input_image, baseline, target_class, noise_std=0.5, num_perturbations=5):
    """
    Performs sensitivity analysis by adding noise to the input and comparing the resulting attributions.

    Args:
        model: The model to evaluate.
        input_image: Input image tensor of shape (1, C, H, W).
        baseline: Baseline image tensor of shape (1, C, H, W).
        target_class: The target class index for which the attributions are computed.
        noise_std: Standard deviation of Gaussian noise to add (set to a high value for this test).
        num_perturbations: Number of perturbed inputs to generate.

    Returns:
        sensitivity_scores: List of similarity scores between original and perturbed attributions.
        all_attributions: List of all perturbed attributions.
        perturbed_predictions: List of model predictions for perturbed inputs.
    """
    # Initialize Integrated Gradients
    ig = IntegratedGradients(model)

    # Compute original attributions
    original_attributions = ig.attribute(input_image, baseline, target=target_class).squeeze().detach().numpy()

    sensitivity_scores = []
    all_attributions = []
    perturbed_predictions = []

    for _ in range(num_perturbations):
        # Add Gaussian noise to the input image (with high noise level)
        noise = torch.randn_like(input_image) * noise_std
        perturbed_input = input_image + noise

        # Compute attributions for the perturbed input
        perturbed_attributions = ig.attribute(perturbed_input, baseline, target=target_class).squeeze().detach().numpy()
        all_attributions.append(perturbed_attributions)

        # Get model prediction for the perturbed input
        perturbed_prediction = torch.argmax(model(perturbed_input), dim=1).item()
        perturbed_predictions.append(perturbed_prediction)

        # Compute similarity between original and perturbed attributions
        similarity, _ = ssim(
            original_attributions,
            perturbed_attributions,
            data_range=original_attributions.max() - original_attributions.min(),
            full=True
        )
        sensitivity_scores.append(similarity)

    return sensitivity_scores, original_attributions, all_attributions, perturbed_predictions


# Function to apply challenging transformations
def apply_challenging_transformations(input_image):
    """
    Apply transformations designed to make the model struggle, including extreme rotations,
    heavy scaling, occlusion, and elastic distortions.

    Args:
        input_image: Input image tensor of shape (1, C, H, W).

    Returns:
        transformed_image: Transformed image tensor.
    """
    # Remove batch dimension
    image = input_image.squeeze(0)  # Now shape is (C, H, W)

    # Apply random extreme rotation
    angle = random.uniform(-90, 90)  # Rotate between -90 and 90 degrees
    image = TF.rotate(image, angle)

    # Apply heavy scaling
    scale = random.uniform(0.5, 1.5)  # Scale between 50% and 150%
    size = [int(image.shape[1] * scale), int(image.shape[2] * scale)]
    image = TF.resize(image, size)

    # Center crop back to original size
    image = TF.center_crop(image, (28, 28))  # Assuming original MNIST size is 28x28

    # Apply random occlusion
    occlusion_size = random.randint(5, 10)  # Occlusion size between 5x5 and 10x10
    x_start = random.randint(0, 28 - occlusion_size)
    y_start = random.randint(0, 28 - occlusion_size)
    image[:, y_start:y_start + occlusion_size, x_start:x_start + occlusion_size] = 0

    # Add batch dimension back
    transformed_image = image.unsqueeze(0)  # Shape is (1, C, H, W)

    return transformed_image



# Function to compute Sensitivity Analysis
def compute_sensitivity(model, input_image, baseline, target_class, num_perturbations=5):
    """
    Performs sensitivity analysis by applying challenging transformations to the input
    and comparing the resulting attributions.

    Args:
        model: The model to evaluate.
        input_image: Input image tensor of shape (1, C, H, W).
        baseline: Baseline image tensor of shape (1, C, H, W).
        target_class: The target class index for which the attributions are computed.
        num_perturbations: Number of perturbed inputs to generate.

    Returns:
        sensitivity_scores: List of similarity scores between original and perturbed attributions.
        all_attributions: List of all perturbed attributions.
        predictions: List of model predictions for perturbed inputs.
    """
    # Initialize Integrated Gradients
    ig = IntegratedGradients(model)

    # Compute original attributions
    original_attributions = ig.attribute(input_image, baseline, target=target_class).squeeze().detach().numpy()

    sensitivity_scores = []
    all_attributions = []
    predictions = []

    for _ in range(num_perturbations):
        # Apply challenging transformations to the input image
        perturbed_input = apply_challenging_transformations(input_image)

        # Compute attributions for the perturbed input
        perturbed_attributions = ig.attribute(perturbed_input, baseline, target=target_class).squeeze().detach().numpy()
        all_attributions.append(perturbed_attributions)

        # Get model prediction for the perturbed input
        perturbed_prediction = torch.argmax(model(perturbed_input), dim=1).item()
        predictions.append(perturbed_prediction)

        # Compute similarity between original and perturbed attributions
        similarity, _ = ssim(
            original_attributions,
            perturbed_attributions,
            data_range=original_attributions.max() - original_attributions.min(),
            full=True
        )
        sensitivity_scores.append(similarity)

    return sensitivity_scores, original_attributions, all_attributions, predictions


## FUNCTIONS FOR LIME 

def predict_function(images, model):
    """
    Prediction function for LIME.
    Takes RGB images, converts them to grayscale, and returns probabilities.
    """

    # Convert RGB (3 channels) to grayscale (1 channel)
    grayscale_images = np.array([rgb2gray(image) for image in images])  # Shape: (batch_size, H, W)

    # Add the channel dimension for PyTorch model compatibility
    grayscale_images = np.expand_dims(grayscale_images, axis=1)  # Shape: (batch_size, 1, H, W)

    # Normalize images
    images_tensor = torch.tensor(grayscale_images).float()
    images_tensor = (images_tensor - 0.1307) / 0.3081  # MNIST normalization

    # Ensure the model is in evaluation mode
    model.eval()

    # Pass through the model
    with torch.no_grad():
        outputs = model(images_tensor)
        logits = outputs[0]

    return logits.softmax(dim=1).numpy()  # Convert logits to probabilities


def predict_on_masked_superpixels(image, model, superpixel_steps=[1, 3, 5, 10, 20]):
    """
    Visualize the impact of masking specific numbers of top superpixels on the image,
     highlighting masked regions.

    Parameters:
        image (numpy.ndarray): Grayscale image to analyze (28x28).
        superpixel_steps (list): List of numbers of top superpixels to mask.

    Returns:
        None
    """
    # Convert grayscale image to RGB
    image_rgb = gray2rgb(image)  # Convert grayscale to RGB (HxWxC)

    # Bind the model to the prediction function
    predict_with_model = partial(predict_function, model=model)


    # Generate LIME explanation
    explanation = explainer.explain_instance(
        image=image_rgb,
        classifier_fn=predict_with_model,
        top_labels=1,
        hide_color=0,
        num_samples=1000,
        segmentation_fn=segmenter
    )

    # Get the predicted label and original confidence
    predicted_label = explanation.top_labels[0]
    original_confidence = predict_function([image_rgb], model)[0, predicted_label]

    # Print original confidence
    print(f"Original Confidence for Predicted Label {predicted_label}: {original_confidence:.2f}")

    # Get the top superpixels ranked by importance
    importance_scores = explanation.local_exp[predicted_label]
    importance_scores.sort(key=lambda x: x[1], reverse=True)  # Sort by importance
    top_superpixels = [idx for idx, _ in importance_scores]

    # Print superpixel importance scores
    print(f"Top Superpixel Importance Scores: {importance_scores[:10]}")  # Show top 10 for brevity

    # Initialize the figure for 2 rows (perturbed image, masked regions)
    num_steps = len(superpixel_steps)
    fig, axes = plt.subplots(2, num_steps, figsize=(4 * num_steps, 8))
    plt.suptitle(f"True Label: {predicted_label}", fontsize=16)

    # Iteratively mask the selected numbers of top superpixels and visualize
    for i, k in enumerate(superpixel_steps):
        perturbed_image = np.copy(image_rgb)
        segments = explanation.segments

        # Create a mask image (red for masked regions)
        mask_visualization = np.copy(image_rgb)

        # Mask the top-k superpixels
        for superpixel in top_superpixels[:k]:
            perturbed_image[segments == superpixel] = 0  # Mask superpixel with black
            mask_visualization[segments == superpixel] = [1, 0, 0]  # Highlight in red

        # Predict confidence and class after perturbation
        perturbed_probabilities = predict_function([perturbed_image], model)[0]
        perturbed_confidence = perturbed_probabilities[predicted_label]
        perturbed_prediction = np.argmax(perturbed_probabilities)

        # Compute confidence drop
        confidence_drop = (original_confidence - perturbed_confidence) / original_confidence

        # Print details for the current step
        print(f"\nTop-{k} Masked:")
        print(f"    Perturbed Confidence: {perturbed_confidence:.2f}")
        print(f"    Confidence Drop: {confidence_drop:.2%}")
        print(f"    Predicted Label After Masking: {perturbed_prediction}")

        # Plot the perturbed image
        axes[0, i].imshow(perturbed_image, interpolation='nearest')
        axes[0, i].set_title(
            f"Top-{k} Masked\nConf Drop: {confidence_drop:.2%}\nPred: {perturbed_prediction}",
            fontsize=10
        )
        axes[0, i].axis('off')

        # Plot the mask visualization
        axes[1, i].imshow(mask_visualization, interpolation='nearest')
        axes[1, i].set_title(f"Masked Superpixels (Top-{k})", fontsize=10)
        axes[1, i].axis('off')

    # Adjust layout
    plt.tight_layout()
    plt.show()
