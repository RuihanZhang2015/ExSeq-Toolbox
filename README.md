
<div align="center">
  <img src="docs/toolbox-logo.png" alt="ExSeq-Toolbox Logo" width="300"/>
</div>

Welcome to ExSeq-Toolbox, a comprehensive Python package for **Expansion Microscopy RNA Sequencing (ExSeq)**. This toolkit enables researchers to create spatially-precise, three-dimensional maps of RNA localization sites within biological tissues at unprecedented resolution.

## Package Overview

ExSeq-Toolbox provides a complete workflow for ExSeq data analysis, from raw image processing to final spatial transcriptomics results. The package is modular, allowing researchers to use individual components or run complete pipelines.

### Core Workflow

1. **Data Preparation & Configuration** - Set up experiment parameters and data organization
2. **Image Alignment** - Register multi-round imaging data with sub-pixel precision
3. **Puncta Detection** - Extract RNA molecules from aligned image volumes
4. **Spatial Analysis** - Map RNA locations and assign gene identities
5. **Visualization** - Generate spatial transcriptomics maps and visualizations

### Key Features

- **Multi-scale alignment** for precise image registration
- **GPU-accelerated processing** for high-throughput analysis
- **Flexible data formats** supporting various microscopy platforms
- **Comprehensive visualization** tools for spatial transcriptomics
- **Modular architecture** enabling custom analysis pipelines

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/RuihanZhang2015/ExSeq-Toolbox.git
cd ExSeq-Toolbox

# Install dependencies
pip install .

# For GPU acceleration (optional)
pip install -r requirements_gpu.txt
```

### Quick Start

```python
from exm.args.args import Args
from exm.align.align import volumetric_alignment
from exm.puncta.extract import extract

# Configure your experiment
args = Args()
args.set_params(
    raw_data_path='/path/to/your/data/',
    spacing=[0.4, 1.625, 1.625],
    channel_names=['640', '594', '561', '488', '405']
)

# Run the complete pipeline
# 1. Align images
volumetric_alignment(args=args)

# 2. Extract RNA puncta
extract(args=args)

# 3. Analyze spatial transcriptomics
# (Additional analysis steps...)
```

## Processing Your Data with Wrapper Scripts

ExSeq-Toolbox provides ready-to-use wrapper scripts that guide you through the complete data processing pipeline. These scripts are located in `examples/wrappers/` and can be customized for your specific experiment.

### 📁 Data Preparation

Before running the wrappers, ensure your data follows this structure:

```
raw_data_directory/
├── code0/
│   ├── raw_fov0.h5
│   ├── raw_fov1.h5
│   ├── raw_fov2.h5
│   └── ...
├── code1/
│   ├── raw_fov0.h5
│   ├── raw_fov1.h5
│   └── ...
└── ...
```

**Important**: Each `raw_fov{}.h5` file should contain datasets named by channel wavelength (e.g., '640', '594', '561', '488', '405').

### 🔧 Step-by-Step Processing

#### 1. Configure Your Experiment
```bash
# Copy and modify the parameter configuration script
cp examples/wrappers/1_pipeline_parameter.py my_experiment_config.py
```

Edit `my_experiment_config.py` to set your parameters:
- **`raw_data_path`**: Path to your raw data directory
- **`spacing`**: Pixel spacing [Z, Y, X] in micrometers
- **`channel_names`**: Your fluorescence channel wavelengths
- **`codes`**: Number of imaging rounds
- **`fovs`**: Fields of view to process

#### 2. Run Image Alignment
```bash
# Copy and modify the alignment script
cp examples/wrappers/2_volume_alignment.py my_alignment.py
```

Edit `my_alignment.py` with your configuration file path, then run:
```bash
python my_alignment.py
```

#### 3. Evaluate Alignment Quality
```bash
# Copy and modify the evaluation script
cp examples/wrappers/3_alignment_evaluation.py my_evaluation.py
```

Run to assess alignment quality and visualize results.

#### 4. Extract RNA Puncta
```bash
# Copy and modify the puncta extraction script
cp examples/wrappers/4_puncta_extraction.py my_puncta_extraction.py
```

Configure GPU/CPU settings and run:
```bash
python my_puncta_extraction.py
```

#### 5. Assign Gene Identities
```bash
# Copy and modify the basecalling script
cp examples/wrappers/5_puncta_basecalling.py my_basecalling.py
```

Run to assign gene identities to detected puncta and map them to nuclei.

### 🚀 Complete Pipeline Example

For a complete workflow, run the scripts in sequence:

```bash
# 1. Set up your experiment parameters
python my_experiment_config.py

# 2. Align your image volumes
python my_alignment.py

# 3. Evaluate alignment quality
python my_evaluation.py

# 4. Extract RNA puncta
python my_puncta_extraction.py

# 5. Assign gene identities
python my_basecalling.py
```

### 📊 Expected Outputs

After running the complete pipeline, you'll have:
- **Aligned image volumes** for each field of view
- **Extracted puncta coordinates** with intensity measurements
- **Gene assignments** for each detected RNA molecule
- **Spatial transcriptomics maps** showing gene expression patterns
- **Quality metrics** and visualization plots

### ⚙️ Customization Tips

- **GPU Acceleration**: Set `use_gpu=True` in puncta extraction for faster processing
- **Memory Management**: Adjust `chunk_size` for large datasets
- **Quality Control**: Modify threshold parameters based on your data quality
- **Parallel Processing**: Increase `parallel_processes` for faster alignment

### 🆘 Troubleshooting

- **Memory Issues**: Reduce `chunk_size` or process fewer FOVs simultaneously
- **Alignment Failures**: Check image quality and adjust alignment parameters
- **Missing Puncta**: Verify channel names and adjust detection thresholds
- **Gene Assignment Errors**: Ensure your gene list CSV is properly formatted

## Documentation

Comprehensive documentation is available at [ExSeq Toolbox Documentation](https://exseq-toolbox.readthedocs.io/en/latest/), including:

- **Installation guides** and system requirements
- **Tutorial notebooks** with example datasets
- **API reference** for all functions and classes
- **Workflow examples** for different research applications
- **Troubleshooting guides** and best practices

## Contributing

We welcome contributions from the research community! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:

- Reporting bugs and feature requests
- Submitting code improvements
- Adding new analysis modules
- Improving documentation

## Citation

If you use ExSeq-Toolbox in your research, please cite:

```
[Citation information to be added]
```

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

## Support

For questions, bug reports, or feature requests, please:

- Open an issue on our [GitHub repository](https://github.com/RuihanZhang2015/ExSeq-Toolbox)
- Check our [documentation](https://exseq-toolbox.readthedocs.io/en/latest/)
- Join our [community discussions](link-to-discussions)

---

