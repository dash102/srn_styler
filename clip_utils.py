from typing import Union, Optional, List, Dict

import numpy as np
import warnings

import os
from PIL import Image
import requests
import torch
import torch.nn as nn
from huggingface_hub import HfFolder, create_repo, get_full_repo_name
from torchvision.transforms import transforms
from transformers import CLIPModel, CLIPConfig, CLIPTokenizer, \
    BatchEncoding, FeatureExtractionMixin, \
    TensorType, BatchFeature, PreTrainedTokenizerBase  # , ProcessorMixin, BatchEncoding, FeatureExtractionMixin, \
#     ImageFeatureExtractionMixin, TensorType, BatchFeature
from transformers.dynamic_module_utils import custom_object_save
from transformers.image_utils import is_torch_tensor, ImageFeatureExtractionMixin
from transformers.processing_utils import AUTO_TO_BASE_CLASS_MAPPING, transformers_module
from transformers.utils import HUGGINGFACE_CO_RESOLVE_ENDPOINT, working_or_temp_dir


import torch
import clip


class CLIPLoss(torch.nn.Module):
    def __init__(self, cache_dir):
        super(CLIPLoss, self).__init__()
        self.model, self.processor, self.tokenizer = get_clip(cache_dir)
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, image, text):
        image = torch.nn.functional.upsample_bilinear(image, (224, 224))
        # similarity = 1 - self.model(image, text)[0] / 100

        inputs = self.processor(text, image, return_tensors="pt", padding=True).to(self.model.device)
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image / 100
        return 1 - logits_per_image

def get_clip(cache_dir, model_name="openai/clip-vit-base-patch32"):
    config = CLIPConfig.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name, cache_dir=cache_dir, config=config)
    processor = CLIPProcessor.from_pretrained(model_name, cache_dir=cache_dir, config=config)
    processor.feature_extractor = CLIPFeatureExtractor(processor.feature_extractor)
    tokenizer = CLIPTokenizer.from_pretrained(model_name, cache_dir=cache_dir, config=config)

    return model, processor, tokenizer


def tensor_to_images(tensor):
    images = []
    for i in range(tensor.shape[0]):
        image = transforms.ToPILImage()(tensor[i].data).convert('RGB')
        images.append(image)
    return images


class StyleScorer(nn.Module):
    def __init__(self, cache_dir, device='cuda:0'):
        super().__init__()
        self.model, self.processor, self.tokenizer = get_clip(cache_dir)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.to(device)

    def forward(self, images, text):
        # images = tensor_to_images(images.cpu())
        inputs = self.processor(text, images, return_tensors="pt", padding=True).to(self.model.device)
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        return 100 - logits_per_image

# images = Image.open('car2.png')
# text = ['a car', 'a plushie car']
# s = StyleScorer('/data/vision/phillipi/fus/clip_models')
# print(s(images, text))

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim

class PushToHubMixin:
    """
    A Mixin containing the functionality to push a model or tokenizer to the hub.
    """

    def _create_repo(
        self,
        repo_id: str,
        private: Optional[bool] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
        repo_url: Optional[str] = None,
        organization: Optional[str] = None,
    ):
        """
        Create the repo if needed, cleans up repo_id with deprecated kwards `repo_url` and `organization`, retrives the
        token.
        """
        if repo_url is not None:
            warnings.warn(
                "The `repo_url` argument is deprecated and will be removed in v5 of Transformers. Use `repo_id` "
                "instead."
            )
            repo_id = repo_url.replace(f"{HUGGINGFACE_CO_RESOLVE_ENDPOINT}/", "")
        if organization is not None:
            warnings.warn(
                "The `organization` argument is deprecated and will be removed in v5 of Transformers. Set your "
                "organization directly in the `repo_id` passed instead (`repo_id={organization}/{model_id}`)."
            )
            if not repo_id.startswith(organization):
                if "/" in repo_id:
                    repo_id = repo_id.split("/")[-1]
                repo_id = f"{organization}/{repo_id}"

        token = HfFolder.get_token() if use_auth_token is True else use_auth_token
        url = create_repo(repo_id=repo_id, token=token, private=private, exist_ok=True)

        # If the namespace is not there, add it or `upload_file` will complain
        if "/" not in repo_id and url != f"{HUGGINGFACE_CO_RESOLVE_ENDPOINT}/{repo_id}":
            repo_id = get_full_repo_name(repo_id, token=token)
        return repo_id, token

    def _get_files_timestamps(self, working_dir: Union[str, os.PathLike]):
        """
        Returns the list of files with their last modification timestamp.
        """
        return {f: os.path.getmtime(os.path.join(working_dir, f)) for f in os.listdir(working_dir)}

    def _upload_modified_files(
        self,
        working_dir: Union[str, os.PathLike],
        repo_id: str,
        files_timestamps: Dict[str, float],
        commit_message: Optional[str] = None,
        token: Optional[str] = None,
        create_pr: bool = False,
    ):
        """
        Uploads all modified files in `working_dir` to `repo_id`, based on `files_timestamps`.
        """
        if commit_message is None:
            if "Model" in self.__class__.__name__:
                commit_message = "Upload model"
            elif "Config" in self.__class__.__name__:
                commit_message = "Upload config"
            elif "Tokenizer" in self.__class__.__name__:
                commit_message = "Upload tokenizer"
            elif "FeatureExtractor" in self.__class__.__name__:
                commit_message = "Upload feature extractor"
            elif "Processor" in self.__class__.__name__:
                commit_message = "Upload processor"
            else:
                commit_message = f"Upload {self.__class__.__name__}"
        modified_files = [
            f
            for f in os.listdir(working_dir)
            if f not in files_timestamps or os.path.getmtime(os.path.join(working_dir, f)) > files_timestamps[f]
        ]
        operations = []
        for file in modified_files:
            operations.append(CommitOperationAdd(path_or_fileobj=os.path.join(working_dir, file), path_in_repo=file))
        logger.info(f"Uploading the following files to {repo_id}: {','.join(modified_files)}")
        return create_commit(
            repo_id=repo_id, operations=operations, commit_message=commit_message, token=token, create_pr=create_pr
        )

    def push_to_hub(
        self,
        repo_id: str,
        use_temp_dir: Optional[bool] = None,
        commit_message: Optional[str] = None,
        private: Optional[bool] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
        max_shard_size: Optional[Union[int, str]] = "10GB",
        create_pr: bool = False,
        **deprecated_kwargs
    ) -> str:
        """
        Upload the {object_files} to the 🤗 Model Hub while synchronizing a local clone of the repo in
        `repo_path_or_name`.

        Parameters:
            repo_id (`str`):
                The name of the repository you want to push your {object} to. It should contain your organization name
                when pushing to a given organization.
            use_temp_dir (`bool`, *optional*):
                Whether or not to use a temporary directory to store the files saved before they are pushed to the Hub.
                Will default to `True` if there is no directory named like `repo_id`, `False` otherwise.
            commit_message (`str`, *optional*):
                Message to commit while pushing. Will default to `"Upload {object}"`.
            private (`bool`, *optional*):
                Whether or not the repository created should be private (requires a paying subscription).
            use_auth_token (`bool` or `str`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `huggingface-cli login` (stored in `~/.huggingface`). Will default to `True` if `repo_url`
                is not specified.
            max_shard_size (`int` or `str`, *optional*, defaults to `"10GB"`):
                Only applicable for models. The maximum size for a checkpoint before being sharded. Checkpoints shard
                will then be each of size lower than this size. If expressed as a string, needs to be digits followed
                by a unit (like `"5MB"`).
            create_pr (`bool`, *optional*, defaults to `False`):
                Whether or not to create a PR with the uploaded files or directly commit.

        Examples:

        ```python
        from transformers import {object_class}

        {object} = {object_class}.from_pretrained("bert-base-cased")

        # Push the {object} to your namespace with the name "my-finetuned-bert".
        {object}.push_to_hub("my-finetuned-bert")

        # Push the {object} to an organization with the name "my-finetuned-bert".
        {object}.push_to_hub("huggingface/my-finetuned-bert")
        ```
        """
        if "repo_path_or_name" in deprecated_kwargs:
            warnings.warn(
                "The `repo_path_or_name` argument is deprecated and will be removed in v5 of Transformers. Use "
                "`repo_id` instead."
            )
            repo_id = deprecated_kwargs.pop("repo_path_or_name")
        # Deprecation warning will be sent after for repo_url and organization
        repo_url = deprecated_kwargs.pop("repo_url", None)
        organization = deprecated_kwargs.pop("organization", None)

        if os.path.isdir(repo_id):
            working_dir = repo_id
            repo_id = repo_id.split(os.path.sep)[-1]
        else:
            working_dir = repo_id.split("/")[-1]

        repo_id, token = self._create_repo(
            repo_id, private=private, use_auth_token=use_auth_token, repo_url=repo_url, organization=organization
        )

        if use_temp_dir is None:
            use_temp_dir = not os.path.isdir(working_dir)

        with working_or_temp_dir(working_dir=working_dir, use_temp_dir=use_temp_dir) as work_dir:
            files_timestamps = self._get_files_timestamps(work_dir)

            # Save all files.
            self.save_pretrained(work_dir, max_shard_size=max_shard_size)

            return self._upload_modified_files(
                work_dir, repo_id, files_timestamps, commit_message=commit_message, token=token, create_pr=create_pr
            )

class ProcessorMixin(PushToHubMixin):
    """
    This is a mixin used to provide saving/loading functionality for all processor classes.
    """

    attributes = ["feature_extractor", "tokenizer"]
    # Names need to be attr_class for attr in attributes
    feature_extractor_class = None
    tokenizer_class = None
    _auto_class = None

    # args have to match the attributes class attribute
    def __init__(self, *args, **kwargs):
        # Sanitize args and kwargs
        for key in kwargs:
            if key not in self.attributes:
                raise TypeError(f"Unexepcted keyword argument {key}.")
        for arg, attribute_name in zip(args, self.attributes):
            if attribute_name in kwargs:
                raise TypeError(f"Got multiple values for argument {attribute_name}.")
            else:
                kwargs[attribute_name] = arg

        if len(kwargs) != len(self.attributes):
            raise ValueError(
                f"This processor requires {len(self.attributes)} arguments: {', '.join(self.attributes)}. Got "
                f"{len(args)} arguments instead."
            )

        # Check each arg is of the proper class (this will also catch a user initializing in the wrong order)
        for attribute_name, arg in kwargs.items():
            class_name = getattr(self, f"{attribute_name}_class")
            # Nothing is ever going to be an instance of "AutoXxx", in that case we check the base class.
            class_name = AUTO_TO_BASE_CLASS_MAPPING.get(class_name, class_name)
            if isinstance(class_name, tuple):
                proper_class = tuple(getattr(transformers_module, n) for n in class_name if n is not None)
            else:
                proper_class = getattr(transformers_module, class_name)

            if not isinstance(arg, proper_class):
                raise ValueError(
                    f"Received a {type(arg).__name__} for argument {attribute_name}, but a {class_name} was expected."
                )

            setattr(self, attribute_name, arg)

    def __repr__(self):
        attributes_repr = [f"- {name}: {repr(getattr(self, name))}" for name in self.attributes]
        attributes_repr = "\n".join(attributes_repr)
        return f"{self.__class__.__name__}:\n{attributes_repr}"

    def save_pretrained(self, save_directory, push_to_hub: bool = False, **kwargs):
        """
        Saves the attributes of this processor (feature extractor, tokenizer...) in the specified directory so that it
        can be reloaded using the [`~ProcessorMixin.from_pretrained`] method.

        <Tip>

        This class method is simply calling [`~feature_extraction_utils.FeatureExtractionMixin.save_pretrained`] and
        [`~tokenization_utils_base.PreTrainedTokenizer.save_pretrained`]. Please refer to the docstrings of the methods
        above for more information.

        </Tip>

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the feature extractor JSON file and the tokenizer files will be saved (directory will
                be created if it does not exist).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs:
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id, token = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)
        # If we have a custom config, we copy the file defining it in the folder and set the attributes so it can be
        # loaded from the Hub.
        if self._auto_class is not None:
            attrs = [getattr(self, attribute_name) for attribute_name in self.attributes]
            configs = [(a.init_kwargs if isinstance(a, PreTrainedTokenizerBase) else a) for a in attrs]
            custom_object_save(self, save_directory, config=configs)

        for attribute_name in self.attributes:
            attribute = getattr(self, attribute_name)
            # Include the processor class in the attribute config so this processor can then be reloaded with the
            # `AutoProcessor` API.
            if hasattr(attribute, "_set_processor_class"):
                attribute._set_processor_class(self.__class__.__name__)
            attribute.save_pretrained(save_directory)

        if self._auto_class is not None:
            # We added an attribute to the init_kwargs of the tokenizers, which needs to be cleaned up.
            for attribute_name in self.attributes:
                attribute = getattr(self, attribute_name)
                if isinstance(attribute, PreTrainedTokenizerBase):
                    del attribute.init_kwargs["auto_map"]

        if push_to_hub:
            self._upload_modified_files(
                save_directory, repo_id, files_timestamps, commit_message=commit_message, token=token
            )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r"""
        Instantiate a processor associated with a pretrained model.

        <Tip>

        This class method is simply calling the feature extractor
        [`~feature_extraction_utils.FeatureExtractionMixin.from_pretrained`] and the tokenizer
        [`~tokenization_utils_base.PreTrainedTokenizer.from_pretrained`] methods. Please refer to the docstrings of the
        methods above for more information.

        </Tip>

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                This can be either:

                - a string, the *model id* of a pretrained feature_extractor hosted inside a model repo on
                  huggingface.co. Valid model ids can be located at the root-level, like `bert-base-uncased`, or
                  namespaced under a user or organization name, like `dbmdz/bert-base-german-cased`.
                - a path to a *directory* containing a feature extractor file saved using the
                  [`~SequenceFeatureExtractor.save_pretrained`] method, e.g., `./my_model_directory/`.
                - a path or url to a saved feature extractor JSON *file*, e.g.,
                  `./my_model_directory/preprocessor_config.json`.
            **kwargs
                Additional keyword arguments passed along to both
                [`~feature_extraction_utils.FeatureExtractionMixin.from_pretrained`] and
                [`~tokenization_utils_base.PreTrainedTokenizer.from_pretrained`].
        """
        args = cls._get_arguments_from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(*args)

    @classmethod
    def register_for_auto_class(cls, auto_class="AutoProcessor"):
        """
        Register this class with a given auto class. This should only be used for custom feature extractors as the ones
        in the library are already mapped with `AutoProcessor`.

        <Tip warning={true}>

        This API is experimental and may have some slight breaking changes in the next releases.

        </Tip>

        Args:
            auto_class (`str` or `type`, *optional*, defaults to `"AutoProcessor"`):
                The auto class to register this new feature extractor with.
        """
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        import transformers.models.auto as auto_module

        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")

        cls._auto_class = auto_class

    @classmethod
    def _get_arguments_from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        args = []
        for attribute_name in cls.attributes:
            class_name = getattr(cls, f"{attribute_name}_class")
            if isinstance(class_name, tuple):
                classes = tuple(getattr(transformers_module, n) if n is not None else None for n in class_name)
                use_fast = kwargs.get("use_fast", True)
                if use_fast and classes[1] is not None:
                    attribute_class = classes[1]
                else:
                    attribute_class = classes[0]
            else:
                attribute_class = getattr(transformers_module, class_name)

            args.append(attribute_class.from_pretrained(pretrained_model_name_or_path, **kwargs))
        return args

class CLIPProcessor(ProcessorMixin):
    r"""
    Constructs a CLIP processor which wraps a CLIP feature extractor and a CLIP tokenizer into a single processor.

    [`CLIPProcessor`] offers all the functionalities of [`CLIPFeatureExtractor`] and [`CLIPTokenizerFast`]. See the
    [`~CLIPProcessor.__call__`] and [`~CLIPProcessor.decode`] for more information.

    Args:
        feature_extractor ([`CLIPFeatureExtractor`]):
            The feature extractor is a required input.
        tokenizer ([`CLIPTokenizerFast`]):
            The tokenizer is a required input.
    """
    feature_extractor_class = "CLIPFeatureExtractor"
    tokenizer_class = ("CLIPTokenizer", "CLIPTokenizerFast")

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)
        self.current_processor = self.feature_extractor

    def __call__(self, text=None, images=None, return_tensors=None, **kwargs):
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to CLIPTokenizerFast's [`~CLIPTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        CLIPFeatureExtractor's [`~CLIPFeatureExtractor.__call__`] if `images` is not `None`. Please refer to the
        doctsring of the above two methods for more information.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.

            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchEncoding`]: A [`BatchEncoding`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """

        if text is None and images is None:
            raise ValueError("You have to specify either text or images. Both cannot be none.")

        if text is not None:
            encoding = self.tokenizer(text, return_tensors=return_tensors, **kwargs)

        if images is not None:
            image_features = self.feature_extractor(images, return_tensors=return_tensors, **kwargs)

        if text is not None and images is not None:
            encoding["pixel_values"] = image_features.pixel_values
            return encoding
        elif text is not None:
            return encoding
        else:
            return BatchEncoding(data=dict(**image_features), tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)


class CLIPFeatureExtractor(FeatureExtractionMixin, ImageFeatureExtractionMixin):
    r"""
    Constructs a CLIP feature extractor.

    This feature extractor inherits from [`FeatureExtractionMixin`] which contains most of the main methods. Users
    should refer to this superclass for more information regarding those methods.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the input to a certain `size`.
        size (`int`, *optional*, defaults to 224):
            Resize the input to the given size. Only has an effect if `do_resize` is set to `True`.
        resample (`int`, *optional*, defaults to `PIL.Image.BICUBIC`):
            An optional resampling filter. This can be one of `PIL.Image.NEAREST`, `PIL.Image.BOX`,
            `PIL.Image.BILINEAR`, `PIL.Image.HAMMING`, `PIL.Image.BICUBIC` or `PIL.Image.LANCZOS`. Only has an effect
            if `do_resize` is set to `True`.
        do_center_crop (`bool`, *optional*, defaults to `True`):
            Whether to crop the input at the center. If the input size is smaller than `crop_size` along any edge, the
            image is padded with 0's and then center cropped.
        crop_size (`int`, *optional*, defaults to 224):
            Desired output size when applying center-cropping. Only has an effect if `do_center_crop` is set to `True`.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether or not to normalize the input with `image_mean` and `image_std`.
        image_mean (`List[int]`, defaults to `[0.485, 0.456, 0.406]`):
            The sequence of means for each channel, to be used when normalizing images.
        image_std (`List[int]`, defaults to `[0.229, 0.224, 0.225]`):
            The sequence of standard deviations for each channel, to be used when normalizing images.
        convert_rgb (`bool`, defaults to `True`):
            Whether or not to convert `PIL.Image.Image` into `RGB` format
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize=True,
        size=224,
        resample=Image.BICUBIC,
        do_center_crop=True,
        crop_size=224,
        do_normalize=True,
        image_mean=None,
        image_std=None,
        do_convert_rgb=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else [0.48145466, 0.4578275, 0.40821073]
        self.image_std = image_std if image_std is not None else [0.26862954, 0.26130258, 0.27577711]
        self.do_convert_rgb = do_convert_rgb

    def __call__(
        self,
        images: Union[
            Image.Image, np.ndarray, "torch.Tensor", List[Image.Image], List[np.ndarray], List["torch.Tensor"]  # noqa
        ],
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several image(s).

        <Tip warning={true}>

        NumPy arrays and PyTorch tensors are converted to PIL images when resizing, so the most efficient is to pass
        PIL images.

        </Tip>

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.

            return_tensors (`str` or [`~utils.TensorType`], *optional*, defaults to `'np'`):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **pixel_values** -- Pixel values to be fed to a model.
        """
        # Input type checking for clearer error
        valid_images = False

        # Check that images has a valid type
        if isinstance(images, (Image.Image, np.ndarray)) or is_torch_tensor(images):
            valid_images = True
        elif isinstance(images, (list, tuple)):
            if len(images) == 0 or isinstance(images[0], (Image.Image, np.ndarray)) or is_torch_tensor(images[0]):
                valid_images = True

        if not valid_images:
            raise ValueError(
                "Images must of type `PIL.Image.Image`, `np.ndarray` or `torch.Tensor` (single example), "
                "`List[PIL.Image.Image]`, `List[np.ndarray]` or `List[torch.Tensor]` (batch of examples)."
            )

        is_batched = bool(
            isinstance(images, (list, tuple))
            and (isinstance(images[0], (Image.Image, np.ndarray)) or is_torch_tensor(images[0]))
        )

        if not is_batched:
            images = [images]

        # transformations (convert rgb + resizing + center cropping + normalization)
        if self.do_convert_rgb:
            images = [self.convert_rgb(image) for image in images]
        if self.do_resize and self.size is not None and self.resample is not None:
            # images = [
            #     self.resize(image=image, size=self.size, resample=self.resample, default_to_square=False)
            #     for image in images
            # ]

            import torch.nn.functional as F
            images = [
                F.interpolate(image, size=(224, 224), mode='bilinear', antialias=True)
                for image in images
            ]

        # if self.do_center_crop and self.crop_size is not None:
        #     images = [center_crop(image, self.crop_size) for image in images]
        if self.do_normalize:
            # images = [self.normalize(image=image, mean=self.image_mean, std=self.image_std) for image in images]
            mean_tensor = torch.Tensor(self.image_mean).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()
            std_tensor = torch.Tensor(self.image_std).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()

            images = [(image - mean_tensor) / std_tensor for image in images]

        # return as BatchFeature
        data = {"pixel_values": images[0]}
        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)

        return encoded_inputs
