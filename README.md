# Foolbox_Chinese

| 文件 | 功能 | 概述 |
| --- | --- | --- |
| `devutils.py `| 包含一些实用函数，如展平张量和扩展张量维度 | `flatten(x: ep.Tensor, keep: int = 1) -> ep.Tensor`：将一个张量`x`展平成一维。可以通过`keep`参数指定从哪个维度开始展平，默认为1。<br/> `atleast_kd(x: ep.Tensor, k: int) -> ep.Tensor`：将一个张量`x`至少扩展到`k`维。通过在维度元组中添加适当数量的1来实现。返回扩展后的张量。 |
| `plot.py` | 绘制图像的模块，提供了将图像数据绘制成图片的函数 |  |
|` types.py` | 定义了几个类型和别名，如Bounds类型和不同距离的别名 |  |
|` tensorboard.py `| 支持将日志记录到TensorBoard的功能 |  |
| `criteria.py `| 定义了一些用于定义对抗性标准的类和函数 | 定义了一些用于定义哪些输入是对抗性的标准。其中包括两个常见的标准：Misclassification 和 TargetedMisclassification。这些标准可以通过子类化Criterion类并实现__call__方法来轻松地实现新的标准。此外，这个模块还定义了一个_And类，用于将两个标准组合成一个新的标准。这个模块还包括一些类型提示和抽象基类的使用。 |
| `distances.py `| 包含用于计算距离的类和函数 | 包含了`Distance`和`LpDistance`两个类。其中`Distance`是一个抽象基类（ABC），而`LpDistance`是`Distance`的子类。<br/>`Distance`类定义了两个抽象方法`__call__`和`clip_perturbation`，分别用于计算参考值和扰动值之间的距离，并将扰动限制在某个范围内。<br/>`LpDistance`类继承自`Distance`类，并实现了父类中定义的两个抽象方法。此类的实例可以用于计算使用Lp范数计算参考值和扰动值之间的距离，并可以将扰动限制在某个范围内。<br/>此外，代码模块还创建了四个`LpDistance`类的实例，分别使用了不同的Lp范数（0、1、2和∞）。在模块的末尾，定义了四个变量`l0`、`l1`、`l2`和`linf`来保存这四个实例。<br/>这个模块的主要功能是提供了计算距离和限制扰动的功能，可用于机器学习中的攻击与防御 |
| `utils.py `| 提供一些实用函数，如计算模型准确率和处理样本 | `accuracy(fmodel, inputs, labels)`：计算给定模型在给定输入和标签上的准确率。<br/> `samples(fmodel, dataset, index, batchsize, shape, data_format, bounds)`：从给定数据集中获取样本，并根据给定的模型、数据格式和范围进行处理。<br/> `_samples(dataset, index, batchsize, shape, data_format, bounds)`：内部函数，从数据集中获取样本和标签。 |
| `gradient_estimators.py` | 用于实现梯度估计的方法 | 一个函数`evolutionary_strategies_gradient_estimator`和一个定义的变量`es_gradient_estimator`。<br/>函数`evolutionary_strategies_gradient_estimator`接受一个`AttackCls`参数，并返回一个具有梯度估计功能的攻击类。函数的参数包括`samples`（样本数量），`sigma`（标准差），`bounds`（范围），和`clip`（是否裁剪）。函数内部首先检查`AttackCls`是否具有`value_and_grad`方法，如果没有，则抛出`ValueError`异常。然后，创建一个名为`GradientEstimator`的新攻击类，继承自`AttackCls`，并重写`value_and_grad`方法来实现梯度估计的逻辑。最后，修改新攻击类的名称和合格名称，并返回它。<br/>变量`es_gradient_estimator`设置为函数`evolutionary_strategies_gradient_estimator`的别名，方便进行调用。 |
| `common.py` | 实用函数，用于处理路径和哈希 | `sha256_hash(git_uri: str) -> str`函数计算给定字符串`git_uri`的SHA256哈希值，并返回哈希值的十六进制表示。 <br/>`home_directory_path(folder: str, hash_digest: str) -> str`函数将给定的文件夹名称和哈希摘要组合成一个路径，并返回完整路径。它使用`os.path`模块来处理路径操作，使用`os.path.join`函数在操作系统上构建路径。 |
| `git_cloner.py `| 用于从远程克隆git仓库的功能 |  |
| `weights_fetcher.py `| 从远程下载并提取模型权重的功能 |  |
| `model_loader.py` | 加载模型的功能和类 |  |
| `zoo.py `| 从Git仓库下载Foolbox兼容模型的功能 |  |
| `deepfool.py` | 实现DeepFool攻击算法 | 包含了DeepFoolAttack类和它的两个子类L2DeepFoolAttack和LinfDeepFoolAttack。DeepFoolAttack是一个简单且快速的基于梯度的对抗攻击方法。它实现了DeepFool攻击算法。L2DeepFoolAttack是DeepFoolAttack的子类，实现了基于L2距离的DeepFool攻击。LinfDeepFoolAttack也是DeepFoolAttack的子类，实现了基于L∞距离的DeepFool攻击。 |
|`carlini_wagner.py`|实现Carlini & Wagner L2攻击算法|定义了一个名为L2CarliniWagnerAttack的类，该类继承自MinimizationAttack类，并实现了run方法。该攻击算法是针对深度神经网络的对抗攻击，其目的是生成对抗样本，使其能够欺骗给定模型。该算法使用二进制搜索来搜索最小的常数，以产生对抗样本。它还使用梯度下降优化器来优化生成对抗样本的过程。该算法的实现参考了Carlini和Wagner的论文[《Towards evaluating the robustness of neural networks》](https://arxiv.org/abs/1608.04644)|
|`newtonfool.py`|实现NewtonFool攻击算法|实现了NewtonFool Attack。它属于Foolbox库中的攻击模块，用于生成针对机器学习模型的对抗样本。使用了NewtonFool算法，并且提供了一些参数来控制攻击的行为，如更新步数和步长。它的主要功能是生成具有干扰的输入样本，以逃避机器学习模型的分类器。|
|`blur.py`|实现高斯模糊攻击算法|用于对输入数据进行高斯模糊攻击，通过增加标准差来模糊化输入数据。攻击类有以下属性和方法：<br />distance: 距离度量对象，可选<br /> steps: 在0到max_sigma之间测试的sigma值的数量，整数，默认值为1000<br /> channel_axis: 输入数据中通道维度的索引值，可选，默认值为None<br />max_sigma: 允许的最大sigma值，可选，默认值为None|
|`spatial_attack.py`|实现旋转和平移攻击算法|基于Madry等人的参考实现（https://github.com/MadryLab/adversarial_spatial）而来，用于生成对抗性的旋转和平移变换。<br/>提供了两种攻击模式：网格搜索和随机搜索。在网格搜索模式下，它将在给定的旋转角度和平移范围内，对输入图像进行多次旋转和平移，以找到能够欺骗给定模型的对抗性样本。在随机搜索模式下，它将随机选择旋转角度和平移范围，并进行多次随机旋转和平移，以找到对抗性样本。<br/>参数: 包括最大平移量、最大旋转角度、平移次数、旋转次数、网格搜索标志和随机搜索步数。<br/>主要的功能方法包括__call__和run。__call__方法将给定的模型、输入图像和准则作为参数，并返回对抗性样本、对抗性样本和成功标志的元组。run方法是执行攻击的核心逻辑，它在给定的模型、输入图像和准则的条件下，进行旋转和平移变换，并返回对抗性样本。repeat方法用于多次重复攻击，但只在随机搜索模式下支持重复。|
|`spatial_attack_transformations.py`|提供图像缩放和重排列的辅助函数||
|`fast_minimum_norm.py`|实现快速最小范数攻击算法|实现快速最小范数攻击。它是基于Lp范数的攻击方法之一，用于在给定步数内找到扰动输入，以欺骗给定的模型。攻击的目标可以是错分类或定向错分类。它使用了一些辅助函数和工具函数，如计算最佳其他类别、计算L1范数球的投影等。这个文件还实现了一个FMNAttackLp基类和一个L1FMNAttack子类，子类通过实现特定的投影和中点方法来扩展基类的功能。|
|`contrast_min.py`|实现对比度减少攻击算法|两个类`BinarySearchContrastReductionAttack`和`LinearSearchContrastReductionAttack`，用于减小输入的对比度以生成最小对抗扰动。<br />`BinarySearchContrastReductionAttack`类使用二分搜索方法来找到最小的对抗扰动。其参数包括：<br />    `distance`：用于搜索最小对抗样本的距离度量。 <br />    `binary_search_steps`：二分搜索的迭代次数，控制结果的精度。<br />    `target`：对比度降低的目标，相对于边界值从0（最小值）到1（最大值）。<br />`LinearSearchContrastReductionAttack`类使用线性搜索方法来找到最小的对抗扰动。其参数包括：<br />    `distance`：用于搜索最小对抗样本的距离度量。<br />    `steps`：线性搜索的步数。 `target`：对比度降低的目标，相对于边界值从0（最小值）到1（最大值）。|
|`gen_attack_utils.py`|提供图像缩放和重排列的工具函数||
|`basic_iterative_method.py`|实现基本迭代方法攻击算法|包括L1 Basic Iterative Method、L2 Basic Iterative Method、L-infinity Basic Iterative Method以及它们分别与Adam optimizer组合的版本。参数包括步长(rel_stepsize)、绝对步长(abs_stepsize)、更新步数(steps)和是否随机起点(random_start)等。每个攻击方法继承自不同的基类(L1BaseGradientDescent、L2BaseGradientDescent、LinfBaseGradientDescent)，其中基类提供了梯度下降的基本操作。|
|`hop_skip_jump.py`|实现HopSkipJump攻击算法|一个无需梯度和概率的强大对抗攻击算法|
|`ddn.py`|实现DDN攻击算法|使用梯度下降的方式进行优化，并在每次迭代中更新扰动向量。攻击的停止条件是达到预定的迭代次数或找到一个成功的对抗样本。可调整的参数: 初始L2范数的值、优化的步数和L2范数的调整因子。|
|`blended_noise.py`|实现混合噪声攻击算法|继承自`FlexibleDistanceMinimizationAttack`类。实现了一个攻击算法，将输入图像与均匀噪声图像混合，直到混合后的图像被错误分类为止。这个攻击算法使用了随机方向和混合步数来搜索最小的对抗性样本。|
|`gradient_descent_base.py`|梯度下降攻击的基类和具体实现||
|`saltandpepper.py`|实现盐和胡椒噪声攻击算法||
|`inversion.py`|实现反转像素值攻击算法||
|`gen_attack.py` | 实现了一个黑盒攻击算法，通过遗传搜索技术生成L-infinity规范的对抗样本。| |
|`base.py` | 实现了攻击的基类和一些派生类，提供了攻击算法的结构和接口。| |
|`pointwise.py` | 实现了一个基于遗传搜索的对抗样本生成算法，逐个维度进行二分搜索。| |
|`projected_gradient_descent.py` | 实现了投影梯度下降攻击算法，包括L1、L2和L无穷范数的版本。| |
|`virtual_adversarial_attack.py` | 实现了一个基于第二阶梯度的对抗样本生成算法，通过近似的二阶优化步骤找到非定向的对抗扰动。| |
|`brendel_bethge.py` | 实现了几种用于最小化攻击的优化方法，用于最小化攻击损失。| |
|`boundary_attack.py` | 实现了边界攻击算法，通过在对抗样本和干净样本之间进行二进制搜索来生成对抗样本。| |
|`additive_noise.py` | 实现了多种基于添加噪声的攻击算法，用于对模型进行攻击。| |
|`dataset_attack.py` | 实现了从给定数据集中选择样本进行攻击的算法，直到找到所有输入的对抗样本。| |
|`contrast.py` | 实现了一种减少对比度的攻击算法，通过添加扰动来降低输入的对比度。| |
|`fast_gradient_method.py` | 实现了快速梯度方法攻击算法，包括L1、L2和L无穷范数的版本。| |
|`binarization.py` | 实现了一个对抗样本生成算法，针对输入进行二值化预处理的模型。| |
|`ead.py` | 实现了一种基于弹性网（Elastic-Net）的对抗样本攻击方法。| |
|`sparse_l1_descent_attack.py` | 实现了一种稀疏L1下降攻击算法，通过稀疏的梯度下降来生成对抗样本。| |
|`clipping_aware_rescaling.py` | 实现了计算批量输入缩放因子的函数。| |
