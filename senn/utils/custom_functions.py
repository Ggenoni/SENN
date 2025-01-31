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

## PLOT MNIST SAMPLES (1 sample from training set, 1 from validation set, 2 from test set)

def plot_samples(train_samples, val_samples, test_samples, title):
    fig, axes = plt.subplots(10, 4, figsize=(4 * 2, 20))  # 10 rows (classes), 4 columns

    fig.suptitle(title, fontsize=15, y=0.92)

    for class_label in range(10):  # Iterate over classes (0-9)
        # Column 1: Training set (1 sample per class)
        img_train = train_samples[class_label][0] if len(train_samples[class_label]) > 0 else None
        if img_train is not None:
            axes[class_label, 0].imshow(img_train.squeeze(), cmap="gray")
        axes[class_label, 0].axis("off")

        # Column 2: Validation set (1 sample per class)
        img_val = val_samples[class_label][0] if len(val_samples[class_label]) > 0 else None
        if img_val is not None:
            axes[class_label, 1].imshow(img_val.squeeze(), cmap="gray")
        axes[class_label, 1].axis("off")

        # Column 3: Test set (1st sample per class)
        img_test1 = test_samples[class_label][0] if len(test_samples[class_label]) > 0 else None
        if img_test1 is not None:
            axes[class_label, 2].imshow(img_test1.squeeze(), cmap="gray")
        axes[class_label, 2].axis("off")

        # Column 4: Test set (2nd sample per class)
        img_test2 = test_samples[class_label][1] if len(test_samples[class_label]) > 1 else None
        if img_test2 is not None:
            axes[class_label, 3].imshow(img_test2.squeeze(), cmap="gray")
        axes[class_label, 3].axis("off")

    # Column titles
    column_titles = ["Train", "Validation", "Test 1", "Test 2"]
    for col in range(4):
        axes[0, col].set_title(column_titles[col], fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Add space for the title
    plt.show()


## FUNCTIONS FOR ABLATION STUDY AND STATS ON SENN

# Ablation study
def evaluate_concept_ablation(model, test_loader, ablated_concept_idx, device='cuda:0'):
    """
    Evaluate the faithfulness of built-in explanations via concept ablation.

    Parameters
    ----------
    model : nn.Module
        The SENN model.
    test_loader : DataLoader
        Dataloader containing test data.
    ablated_concept_idx : list[int]
        List of indices of concepts to ablate (set to zero).
    device : str
        The device on which to run the evaluation (e.g., 'cuda:0' or 'cpu').

    Returns
    -------
    original_predictions : list
        Predictions made by the model without ablation.
    ablated_predictions : list
        Predictions made by the model after concept ablation.
    impact_on_prediction : list
        Boolean list indicating whether the prediction changed after ablation.
    """
    model.eval()  # Set model to evaluation mode
    original_predictions = []
    ablated_predictions = []
    impact_on_prediction = []

    # Iterate over the test data
    for test_batch, _ in test_loader:
        test_batch = test_batch.to(device).float()

        # Get original predictions and concepts
        with torch.no_grad():
            original_preds, (concepts, relevances), _ = model(test_batch)
            original_preds = original_preds.argmax(dim=1)

        # Ablate the selected concepts by setting them to zero
        concepts_ablation = concepts.clone()
        concepts_ablation[:, ablated_concept_idx, :] = 0  # Set specific concepts to 0

        # Recompute predictions with ablated concepts
        with torch.no_grad():
            ablated_preds = model.aggregator(concepts_ablation, relevances)
            ablated_preds = ablated_preds.argmax(dim=1)

        # Compare original and ablated predictions
        for orig, ablated in zip(original_preds, ablated_preds):
            original_predictions.append(orig.item())
            ablated_predictions.append(ablated.item())
            impact_on_prediction.append(orig.item() != ablated.item())  # Check if prediction changed

    return original_predictions, ablated_predictions, impact_on_prediction

# Relevance scores

def analyze_relevance_scores_by_predicted_class(model, test_loader, num_concepts, device='cuda:0'):
    """
    Analyze relevance scores (theta) across the dataset for the predicted class.

    Parameters
    ----------
    model : nn.Module
        The trained SENN model.
    test_loader : DataLoader
        The test data loader.
    num_concepts : int
        Total number of concepts in the model.
    device : str
        Device to run the analysis on ('cuda:0' or 'cpu').

    Returns
    -------
    relevance_stats : dict
        A dictionary containing average and standard deviation of relevance scores for each concept.
    """
    model.eval()
    model.to(device)

    relevance_scores = []

    with torch.no_grad():
        for test_batch, _ in test_loader:
            test_batch = test_batch.to(device).float()

            # Forward pass to get predictions and relevance scores
            y_pred, (_, relevances), _ = model(test_batch)
            y_pred = y_pred.argmax(dim=1)  # Shape: (batch_size,)

            # Gather relevance scores only for the predicted class
            batch_relevances = relevances[torch.arange(relevances.size(0)), :, y_pred].cpu().numpy()
            relevance_scores.append(batch_relevances)

    # Combine all batches into a single array
    relevance_scores = np.concatenate(relevance_scores, axis=0)  # Shape: (num_samples, num_concepts)

    # Calculate statistics for each concept
    relevance_stats = {
        f"{i}": {
            "mean": np.mean(relevance_scores[:, i]),
            "std": np.std(relevance_scores[:, i])
        }
        for i in range(num_concepts)
    }

    # Visualize relevance scores
    concept_means = [stat["mean"] for stat in relevance_stats.values()]
    concept_stds = [stat["std"] for stat in relevance_stats.values()]

    plt.figure(figsize=(10, 6))
    x = np.arange(num_concepts)

    # Plot bar chart with error bars
    plt.bar(x, concept_means, yerr=concept_stds, capsize=5, alpha=0.7, color='blue', label="Mean Relevance")
    plt.xticks(x, [f"Concept {i+1}" for i in range(num_concepts)])
    plt.xlabel("Concepts")
    plt.ylabel("Relevance Scores (Predicted Class)")
    plt.title("Relevance Scores (Theta) Across the Dataset (Predicted Class Only)")
    plt.legend()
    plt.show()

    return relevance_stats


# Class specific relevance

def analyze_class_specific_relevance(model, test_loader, num_classes, num_concepts, device='cuda:0'):
    """
    Analyze relevance scores (theta) by class across the dataset.

    Parameters
    ----------
    model : nn.Module
        The trained SENN model.
    test_loader : DataLoader
        The test data loader.
    num_classes : int
        Total number of classes in the dataset (e.g., 10 for MNIST).
    num_concepts : int
        Total number of concepts in the model.
    device : str
        Device to run the analysis on ('cuda:0' or 'cpu').

    Returns
    -------
    class_relevance_stats : dict
        A dictionary containing the mean and standard deviation of relevance scores for each concept, grouped by class.
    """
    model.eval()
    model.to(device)

    # Initialize containers to store relevance scores by class
    class_relevance_scores = {cls: [] for cls in range(num_classes)}

    with torch.no_grad():
        for test_batch, test_labels in test_loader:
            test_batch = test_batch.to(device).float()
            test_labels = test_labels.to(device)

            # Forward pass to get predictions and relevance scores
            y_pred, (_, relevances), _ = model(test_batch)
            y_pred = y_pred.argmax(dim=1)  # Shape: (batch_size,)

            # Gather relevance scores for the predicted class
            batch_relevances = relevances[torch.arange(relevances.size(0)), :, y_pred].cpu().numpy()

            # Append relevance scores to the corresponding class
            for cls in range(num_classes):
                class_relevance_scores[cls].append(batch_relevances[test_labels.cpu().numpy() == cls])

    # Compute mean and standard deviation of relevance scores for each class and concept
    class_relevance_stats = {}
    for cls, relevances in class_relevance_scores.items():
        if len(relevances) > 0:
            relevances = np.concatenate(relevances, axis=0)  # Combine all batches
            class_relevance_stats[cls] = {
                f"Concept {i+1}": {
                    "mean": np.mean(relevances[:, i]),
                    "std": np.std(relevances[:, i])
                }
                for i in range(num_concepts)
            }

    # Visualize relevance scores by class
    for cls, stats in class_relevance_stats.items():
        concept_means = [stat["mean"] for stat in stats.values()]
        concept_stds = [stat["std"] for stat in stats.values()]

        plt.figure(figsize=(10, 6))
        x = np.arange(num_concepts)

        # Plot bar chart with error bars
        plt.bar(x, concept_means, yerr=concept_stds, capsize=5, alpha=0.7, color='blue', label=f"Class {cls}")
        plt.xticks(x, [f"Concept {i+1}" for i in range(num_concepts)])
        plt.xlabel("Concepts")
        plt.ylabel("Relevance Scores")
        plt.title(f"Relevance Scores (Theta) for Class {cls}")
        plt.legend()
        plt.show()

    return class_relevance_stats




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
def sensitivity_analysis(model, input_image, baseline, target_class, challenging_transformations=False, noise_std=0.5, num_perturbations=5):
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

        if challenging_transformations:
            # Apply challenging transformations to the input image
            perturbed_input = apply_challenging_transformations(input_image)
        else:
            # Add Gaussian noise to the input image (with high noise level)
            noise = torch.randn_like(input_image) * noise_std
            perturbed_input = input_image + noise

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


# Masking relevan pixels
def predict_on_masked_pixels(image, model, ig_attributions, predicted_label, pixel_steps=[10, 50, 100, 200, 500]):
    """
    Visualize the impact of masking top important pixels (IG-based) on confidence.

    Parameters:
        image (numpy.ndarray): 2D grayscale image (28x28).
        model (torch.nn.Module): The trained model.
        ig_attributions (numpy.ndarray): Integrated Gradients attributions (28x28).
        predicted_label (int): The model’s predicted class.
        pixel_steps (list): List of numbers of top pixels to mask.

    Returns:
        None
    """
    # Convert image to float and normalize
    image = np.copy(image)  # Prevent modifying the original data

    # Convert grayscale image to RGB for visualization
    image_rgb = gray2rgb(image)

    # Flatten IG attributions and get indices of top pixels
    ig_flat = ig_attributions.flatten()
    pixel_ranks = np.argsort(-np.abs(ig_flat))  # Sort by absolute importance

    # Predict function
    predict_with_model = partial(predict_function, model=model)

    # Get original confidence
    original_confidence = predict_function([image_rgb], model)[0, predicted_label]
    print(f"Original Confidence for Predicted Label {predicted_label}: {original_confidence:.2f}")

    # Plot results
    num_steps = len(pixel_steps)
    fig, axes = plt.subplots(2, num_steps, figsize=(4 * num_steps, 8))
    plt.suptitle(f"True Label: {predicted_label}", fontsize=16)

    # Iterate over pixel removal steps
    for i, k in enumerate(pixel_steps):
        perturbed_image = np.copy(image)  # Copy to prevent modifying original
        mask_visualization = np.copy(image_rgb)

        # Mask the top-k most important pixels
        for pixel_idx in pixel_ranks[:k]:
            row, col = np.unravel_index(pixel_idx, image.shape)
            perturbed_image[row, col] = 0  # Mask pixel with black
            mask_visualization[row, col] = [1, 0, 0]  # Highlight in red

        # Predict on the masked image
        perturbed_probabilities = predict_function([gray2rgb(perturbed_image)], model)[0]
        perturbed_confidence = perturbed_probabilities[predicted_label]
        perturbed_prediction = np.argmax(perturbed_probabilities)

        # Compute confidence drop
        confidence_drop = (original_confidence - perturbed_confidence) / original_confidence

        # Print results
        print(f"\nTop-{k} Pixels Masked:")
        print(f"    Perturbed Confidence: {perturbed_confidence:.2f}")
        print(f"    Confidence Drop: {confidence_drop:.2%}")
        print(f"    Predicted Label After Masking: {perturbed_prediction}")

        # Plot the perturbed image
        axes[0, i].imshow(perturbed_image, cmap="gray", interpolation="nearest")
        axes[0, i].set_title(
            f"Top-{k} Masked\nConf Drop: {confidence_drop:.2%}\nPred: {perturbed_prediction}",
            fontsize=10
        )
        axes[0, i].axis("off")

        # Plot the mask visualization
        axes[1, i].imshow(mask_visualization, interpolation="nearest")
        axes[1, i].set_title(f"Masked Pixels (Top-{k})", fontsize=10)
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()

## FUNCTIONS FOR LIME 

# ==> CHOOSE THE BEST SEGMENTATION ALGORITHM <==
segmenter = SegmentationAlgorithm('slic', n_segments=100, compactness=10)
# segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=150, ratio=0.2)
# segmenter = SegmentationAlgorithm('felzenszwalb', scale=100, sigma=0.8, min_size=5)

# Initialize the LIME Image Explainer
explainer = lime_image.LimeImageExplainer()


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
