import torch
import torch.nn as nn
import torch.nn.functional as F


class SENN(nn.Module):
    def __init__(self, conceptizer, parameterizer, aggregator):
        """Represents a Self Explaining Neural Network (SENN).
        (https://papers.nips.cc/paper/8003-towards-robust-interpretability-with-self-explaining-neural-networks)

        A SENN model is a neural network made explainable by design. It is made out of several submodules:
            - conceptizer
                Model that encodes raw input into interpretable feature representations of
                that input. These feature representations are called concepts.
            - parameterizer
                Model that computes the parameters theta from given the input. Each concept
                has with it associated one theta, which acts as a ``relevance score'' for that concept.
            - aggregator
                Predictions are made with a function g(theta_1 * h_1, ..., theta_n * h_n), where
                h_i represents concept i. The aggregator defines the function g, i.e. how each
                concept with its relevance score is combined into a prediction.

        Parameters
        ----------
        conceptizer : Pytorch Module
            Model that encodes raw input into interpretable feature representations of
            that input. These feature representations are called concepts.

        parameterizer : Pytorch Module
            Model that computes the parameters theta from given the input. Each concept
            has with it associated one theta, which acts as a ``relevance score'' for that concept.

        aggregator : Pytorch Module
            Predictions are made with a function g(theta_1 * h_1, ..., theta_n * h_n), where
            h_i represents concept i. The aggregator defines the function g, i.e. how each
            concept with its relevance score is combined into a prediction.
        """
        super().__init__()
        self.conceptizer = conceptizer
        self.parameterizer = parameterizer
        self.aggregator = aggregator

    def forward(self, x):
        """Forward pass of SENN module.
        
        In the forward pass, concepts and their reconstructions are created from the input x.
        The relevance parameters theta are also computed.

        Parameters
        ----------
        x : torch.Tensor
            Input data tensor of shape (BATCH, *). Only restriction on the shape is that
            the first dimension should correspond to the batch size.

        Returns
        -------
        predictions : torch.Tensor
            Predictions generated by model. Of shape (BATCH, *).
            
        explanations : tuple
            Model explanations given by a tuple (concepts, relevances).

            concepts : torch.Tensor
                Interpretable feature representations of input. Of shape (NUM_CONCEPTS, *).

            parameters : torch.Tensor
                Relevance scores associated with concepts. Of shape (NUM_CONCEPTS, *)
        """
        concepts, recon_x = self.conceptizer(x)
        relevances = self.parameterizer(x)
        predictions = self.aggregator(concepts, relevances)
        explanations = (concepts, relevances)
        return predictions, explanations, recon_x


class DiSENN(nn.Module):
    """Self-Explaining Neural Network with Disentanglement 

    DiSENN is an extension of the Self-Explaining Neural Network proposed by [1]
    
    DiSENN incorporates a constrained variational inference framework on a 
    SENN Concept Encoder to learn disentangled representations of the 
    basis concepts as in [2]. The basis concepts are then independently
    sensitive to single generative factors leading to better interpretability 
    and lesser overlap with other basis concepts. Such a strong constraint 
    better fulfills the "diversity" desiderata for basis concepts
    in a Self-Explaining Neural Network.

    References
    ----------
    [1] Alvarez Melis, et al.
    "Towards Robust Interpretability with Self-Explaining Neural Networks" NIPS 2018
    [2] Irina Higgins, et al. 
    ”β-VAE: Learning basic visual concepts with a constrained variational framework.” ICLR 2017. 
    
    """
    
    def __init__(self, vae_conceptizer, parameterizer, aggregator):
        """Instantiates the SENDD with a variational conceptizer, parameterizer and aggregator

        Parameters
        ----------
        vae_conceptizer : nn.Module
            A variational inference model that learns a disentangled distribution over
            the prior basis concepts given the input posterior.

        parameterizer : nn.Module
            Model that computes the parameters theta from given the input. Each concept
            has with it associated one theta, which acts as a ``relevance score'' for that concept.

        aggregator : nn.Module
            Predictions are made with a function g(theta_1 * h_1, ..., theta_n * h_n), where
            h_i represents concept i. The aggregator defines the function g, i.e. how each
            concept with its relevance score is combined into a prediction.
        """
        super().__init__()
        self.vae_conceptizer = vae_conceptizer
        self.parameterizer = parameterizer
        self.aggregator = aggregator

    def forward(self, x):
        """Forward pass of a DiSENN model
        
        The forward pass computes a distribution over basis concepts
        and the corresponding relevance scores. The mean concepts 
        and relevance scores are aggregated to generate a prediction.
        
        Parameters
        ----------
        x : torch.Tensor
            Input data tensor of shape [batch_size, ...]
            
        Returns
        -------
        predictions : torch.Tensor
            Predictions generated by the DiSENN model of shape [batch_size, ...]

        explanations : tuple
            Explanation give by the model as a nested tuple of 
            relevance scores and concept distribution as mean and log variance:
            ((concept_mean, concept_log_variance), relevance_score)

            concept_mean : torch.Tensor
                Mean of the disentangled concept distribution of shape
                [batch_size, num_concepts, concept_dim]

            concept_log_varance : torch.Tensor
                Log Variance of the disentangled concept distribution of shape
                [batch_size, num_concepts, concept_dim]

            relevance_score : torch.Tensor
                Relevance scores (for each concept and class) of shape 
                [batch_size, num_concepts, num_classes]
        """
        concept_mean, concept_logvar, x_reconstruct = self.vae_conceptizer(x)
        relevances = self.parameterizer(x)
        predictions = self.aggregator(concept_mean, relevances)
        explanations = ((concept_mean, concept_logvar), relevances)
        return predictions, explanations, x_reconstruct
    
    def explain(self, x, num_prototypes=20, traversal_range=2, show=False, save=False, save_as=None):
        """Explains the model predictions of input x"""

        assert len(x.shape) == 4, \
        "input x must be a rank 4 tensor of shape batch_size x channel x width x height"
        
        y_pred, explanations, x_reconstruct = model(x)
        (x_posterior_mean, x_posterior_logvar), relevances = explanations
        x_posterior_mean = x_posterior_mean.squeeze(-1)
        x_posterior_logvar = x_posterior_logvar.squeeze(-1)
        
        concepts = x_posterior_mean.detach().numpy()
        num_concepts = concepts.shape[1]
        concepts_sample = model.vae_conceptizer.sample(x_posterior_mean,
                                                    x_posterior_logvar).detach()
        # generate new concept vector for each prototype
        # by traversing independently in each dimension
        concepts_sample = concepts_sample.repeat(num_prototypes, 1)
        concepts_traversals = [independent_traversal(concepts_sample, dim,\
                                                    traversal_range, num_prototypes) 
                            for dim in range(num_concepts)]
        concepts_traversals = torch.cat(concepts_traversals, dim=0)
        prototypes = model.vae_conceptizer.decoder(concepts_traversals)
        prototype_imgs = prototypes.view(-1, x.shape[1], x.shape[2], x.shape[3])
        
        # nrow is number of images in a row which must be the number of prototypes
        prototype_grid_img = make_grid(prototype_imgs, nrow=num_prototypes).detach().numpy()
        
        # prepare to plot
        relevances = relevances.squeeze(0).detach().numpy()
        relevances = relevances[:,y_pred.argmax(1)]
        concepts = concepts.squeeze(0)
        relevances_colors = ['g' if r > 0 else 'r' for r in relevances]
        concepts_colors = ['g' if c > 0 else 'r' for c in concepts]
        
        # plot input image, relevances, concepts, prototypes side by side
        plt.style.use('seaborn-paper')
        gridsize = (1, 6)
        fig = plt.figure(figsize=(18,3))
        ax1 = plt.subplot2grid(gridsize, (0,0))
        ax2 = plt.subplot2grid(gridsize, (0,1))
        ax3 = plt.subplot2grid(gridsize, (0,2))
        ax4 = plt.subplot2grid(gridsize, (0,3), colspan=3)

        ax1.imshow(x.numpy().squeeze(), cmap='gray')
        ax1.set_axis_off()
        ax1.set_title(f'Input Prediction: {y_pred.argmax(1).item()}', fontsize=18)

        ax2.barh(range(num_concepts), relevances, color=relevances_colors)
        ax2.set_xlabel('Relevance Scores', fontsize=18)
        ax2.xaxis.set_label_position('top')
        ax2.tick_params(axis='x', which='major', labelsize=12)
        ax2.set_yticks([])

        ax3.barh(range(num_concepts), concepts, color=concepts_colors)
        ax3.set_xlabel('Concepts', fontsize=18)
        ax3.xaxis.set_label_position('top')
        ax3.tick_params(axis='x', which='major', labelsize=12)
        ax3.set_yticks([])

        ax4.imshow(prototype_grid_img.transpose(1,2,0))
        ax4.set_title('Prototypes', fontsize=18)
        ax4.set_axis_off()

        fig.tight_layout()

        if show:
            fig.show()
        if save:
            plt.savefig(save_as)
            plt.clf()        