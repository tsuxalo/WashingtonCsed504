# Contributing to UW CSED 504

Thank you for contributing to the shared class repository! Please follow these guidelines to keep the repository organized and useful for everyone.

## Branching

- The `main` branch contains reviewed, instructor-approved content.
- Create a new branch for your work:
  ```bash
  git checkout -b your-uw-netid/feature-description
  ```

## Commit Messages

Write short, descriptive commit messages in the imperative mood:

- ✅ `Add starter code for Assignment 2`
- ✅ `Fix typo in lab 3 instructions`
- ❌ `stuff`
- ❌ `fixed things`

## Directory Guidelines

| Directory      | Contents                                                 |
|----------------|----------------------------------------------------------|
| `assignments/` | Homework starter code, problem statements, and rubrics   |
| `labs/`        | In-class lab exercises and activity files                |
| `resources/`   | Supplementary references, slides, and reading links      |
| `projects/`    | Course project templates, guidelines, and sample code    |

## Pull Requests

1. Push your branch to the remote:
   ```bash
   git push origin your-uw-netid/feature-description
   ```
2. Open a Pull Request on GitHub against `main`.
3. Fill in the PR description and tag a classmate or the instructor for review.

## Code Style

This course uses Python. Please follow [PEP 8](https://peps.python.org/pep-0008/) style guidelines. You can auto-format your code with:

```bash
pip install ruff
ruff format .
ruff check .
```

## Questions

Post questions in the course discussion board or open a GitHub Issue in this repository.
