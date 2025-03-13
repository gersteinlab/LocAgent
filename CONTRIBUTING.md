# Contributing to *PLACEHOLDER*

We welcome contributions from everyone to help improve and expand *PLACEHOLDER*. This document outlines the process for contributing to the project.

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Coding Standards](#coding-standards)
3. [Pull Request Process](#pull-request-process)
4. [Pull Request Template](#pull-request-template)

## Environment Setup

To contribute to *PLACEHOLDER*, follow these steps to set up your development environment:

1. Clone the repository:
   ```
   git clone https://github.com/gersteinlab/placeholder.git
   cd placeholder
   ```
2. Create a Conda environment:
   ```
   conda create -n placeholder python=3.10
   conda activate placeholder
   ```
3. Install the project in editable mode with development dependencies:
   ```
   python3 -m pip install --upgrade pip
   pip install -e .
   ```

## Coding Standards

We strive to maintain clean and consistent code throughout the project. Please adhere to the following guidelines:

1. Follow PEP 8 guidelines for Python code.
2. Use meaningful variable and function names.
3. Write docstrings for functions and classes.
4. Keep functions small and focused on a single task.
5. Use type hints where appropriate.

### Code Formatting

We use `black` for code formatting. To ensure your code is properly formatted:

1. Install black:
   ```
   pip install black
   ```
2. Run black on the codebase:
   ```
   black .
   ```

## Pull Request Process

1. Create a new branch for your feature or bugfix; feature is for new function; bugfix is for fixing a bug:
   ```
   git checkout -b feature/your-feature-name
   ```
2. Make your changes and commit them with clear, concise commit messages.
   1. Monitor the current conditions and check which files are modified or untracked
   ```
   git status
   ```
   2. Git add your file
   ```
   git add schema.py 
   ```
   3. Submit your change and commit
   ```
   git commit -m "message"
   ```
4. Push your branch to the repository:
   ```
   git push origin feature/your-feature-name
   ```
5. Open a pull request against the `main` branch on the website.
6. Fill out the pull request template (see below).
7. Address any feedback or comments from reviewers.

## Pull Request Template

When you open a new pull request, please use the following template:

```markdown
## Description

### Changes
[Provide a detailed list of the changes made in this PR]

### Design
[Explain the design decisions and architectural changes, if any]

### Example Code
[If applicable, provide example code demonstrating the usage of new features or fixes]

## Related Issue
[Link to the issue this PR addresses, if applicable]

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] This change requires a documentation update

## How Has This Been Tested?
[Describe the tests you ran to verify your changes]

## Additional Notes
[Add any additional information or context about the PR here]
```

Thank you for contributing to *PLACEHOLDER*!
