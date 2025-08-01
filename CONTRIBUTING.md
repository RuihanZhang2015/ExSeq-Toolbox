# Contributing to ExSeq-Toolbox

Thank you for your interest in contributing to ExSeq-Toolbox! We welcome contributions from researchers, developers, and the broader scientific community. This document provides guidelines to help you get started.

## How to Contribute

### 🐛 Reporting Bugs

If you find a bug, please:

1. **Check existing issues** - Search the [GitHub Issues](https://github.com/RuihanZhang2015/ExSeq-Toolbox/issues) to see if the bug has already been reported
2. **Create a new issue** with:
   - A clear, descriptive title
   - Steps to reproduce the bug
   - Expected vs. actual behavior
   - Your system information (OS, Python version, etc.)
   - Any error messages or logs

### 💡 Suggesting Features

We welcome feature suggestions! Please:

1. **Search existing issues** to avoid duplicates
2. **Create a feature request** with:
   - A clear description of the feature
   - Use cases and benefits
   - Any relevant research context
   - Implementation ideas (if you have them)

### 🔧 Code Contributions

#### Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/ExSeq-Toolbox.git
   cd ExSeq-Toolbox
   ```
3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements_gpu.txt  # If using GPU features
   ```
5. **Install in development mode**:
   ```bash
   pip install -e .
   ```

#### Development Workflow

1. **Create a new branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. **Make your changes** following our coding standards

3. **Test your changes**:
   ```bash
   # Run existing tests
   pytest tests/
   
   # Run with coverage
   pytest --cov=exm tests/
   ```

4. **Update documentation** if needed

5. **Commit your changes** with clear commit messages:
   ```bash
   git commit -m "Add feature: brief description of changes"
   ```

6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request** with:
   - A clear title and description
   - Reference to any related issues
   - Summary of changes
   - Any new dependencies or requirements

## Coding Standards

### Python Code Style

- Follow **PEP 8** style guidelines
- Use **type hints** for function parameters and return values
- Write **docstrings** for all public functions and classes
- Keep functions focused and **under 50 lines** when possible
- Use **descriptive variable names**

### Documentation

- **Update docstrings** for any modified functions
- **Add examples** in docstrings for complex functions
- **Update README.md** if adding new features
- **Update API documentation** if changing public interfaces

### Testing

- **Write tests** for new functionality
- **Ensure existing tests pass** before submitting
- **Add integration tests** for new modules
- **Test edge cases** and error conditions

## Areas for Contribution

### High Priority

- **Bug fixes** and performance improvements
- **Documentation improvements** and examples
- **Test coverage** improvements
- **GPU optimization** for existing functions

### Medium Priority

- **New analysis modules** for spatial transcriptomics
- **Visualization tools** for ExSeq data
- **Data format support** for additional microscopy platforms
- **Workflow automation** and pipeline tools

### Low Priority

- **UI improvements** for existing tools
- **Additional file format support**
- **Performance optimizations** for specific use cases

## Review Process

1. **Automated checks** will run on your PR (tests, linting, etc.)
2. **Maintainers will review** your code and provide feedback
3. **Address any feedback** and update your PR
4. **Once approved**, your PR will be merged

## Getting Help

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Documentation**: Check our [docs](https://exseq-toolbox.readthedocs.io/en/latest/)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please:

- **Be respectful** and constructive in all interactions
- **Help newcomers** learn and contribute
- **Give credit** to others for their contributions
- **Focus on the science** and technical merit

## Recognition

Contributors will be recognized in:
- **GitHub contributors list**
- **Release notes** for significant contributions
- **Documentation** where appropriate

---

Thank you for contributing to ExSeq-Toolbox!