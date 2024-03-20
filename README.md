MSC Thesis

### Known Issue with gym 0.21.0

If you encounter issues installing `gym-retro` with `gym` version 0.21.0, you may come across a bug related to TensorFlow dependencies. The issue arises due to conflicting dependencies between `gym` and TensorFlow.

However, this issue can be resolved by following the workaround provided [here](https://github.com/openai/gym/issues/3202).

"pip install git+https://github.com/openai/gym.git@9180d12e1b66e7e2a1a622614f787a6ec147ac40"

installs gym==0.21 with the typo opencv-python>=3. fixed.

Please ensure to follow the steps mentioned in the linked GitHub issue to successfully install `gym-retro` with `gym` version 0.21.0.
