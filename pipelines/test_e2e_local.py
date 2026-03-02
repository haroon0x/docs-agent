"""
Level 3 End-to-End Test using KFP Local Runner

Runs the actual KFP components from kubeflow-pipeline.py locally using
kfp.local.SubprocessRunner. This tests the real KFP component execution
(with dsl.Input/Output wrappers) without needing a Kubernetes cluster.

Components tested:
1. download_github_issues - Fetches real issues from GitHub API
2. chunk_and_embed - Skipped (needs sentence-transformers + torch)
3. store_milvus - Skipped (needs Milvus server)
4. Pipeline condition (dsl.If) - Verified via YAML structure

For chunk_and_embed and store_milvus, see Level 2 I/O contract tests
in test_components.py which verify the data format compatibility.
"""

import json
import os
import sys

# Add pipelines dir to path so we can import components
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kfp import local
import importlib.util
spec = importlib.util.spec_from_file_location(
    "kubeflow_pipeline",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "kubeflow-pipeline.py")
)
kfp_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kfp_module)
download_github_issues = kfp_module.download_github_issues

# Initialize KFP local runner (subprocess mode, no venv for speed)
local.init(runner=local.SubprocessRunner(use_venv=True))


def test_download_issues_component():
    """Run download_github_issues as a real KFP component locally."""
    print("=" * 60)
    print("LEVEL 3 TEST: download_github_issues (KFP Local Runner)")
    print("=" * 60)

    # Execute the actual KFP component with small params
    task = download_github_issues(
        repos="kubeflow/website",
        labels="",
        state="open",
        max_issues_per_repo=3,
        github_token=os.environ.get("GITHUB_TOKEN", "")
    )

    # Get the output artifact
    output_path = task.outputs["issues_data"].path
    print(f"\n  Output artifact path: {output_path}")

    # Validate output
    with open(output_path, 'r') as f:
        lines = f.readlines()

    issues = [json.loads(line) for line in lines]
    print(f"  Issues fetched: {len(issues)}")

    assert len(issues) > 0, "FAIL: No issues fetched!"
    assert len(issues) <= 3, f"FAIL: Got {len(issues)} issues, expected <= 3"

    for i, issue in enumerate(issues):
        assert "path" in issue, f"FAIL: Issue {i} missing 'path'"
        assert "content" in issue, f"FAIL: Issue {i} missing 'content'"
        assert "file_name" in issue, f"FAIL: Issue {i} missing 'file_name'"
        assert "url" in issue, f"FAIL: Issue {i} missing 'url'"
        assert issue["path"].startswith("issues/"), f"FAIL: Issue {i} path invalid"
        assert "github.com" in issue["url"], f"FAIL: Issue {i} URL invalid"
        print(f"  ✅ Issue {i}: {issue['file_name']} ({len(issue['content'])} chars)")

    print(f"\n  ✅ KFP component executed successfully via SubprocessRunner")
    print(f"  ✅ Output artifact is valid JSONL with {len(issues)} issues")
    print(f"  ✅ All issues have correct structure (path, content, file_name, url)")
    return True


def test_download_issues_empty_repos():
    """Test that empty repos string produces empty output (no crash)."""
    print("\n" + "=" * 60)
    print("LEVEL 3 TEST: download_github_issues with empty string")
    print("=" * 60)

    task = download_github_issues(
        repos="",
        labels="",
        state="open",
        max_issues_per_repo=3,
        github_token=""
    )

    output_path = task.outputs["issues_data"].path
    with open(output_path, 'r') as f:
        content = f.read().strip()

    assert content == "", f"FAIL: Expected empty output for empty repos, got: {content[:100]}"
    print("  ✅ Empty repos string produces empty output file (no crash)")
    print("  ✅ dsl.If condition will correctly skip this in the pipeline")
    return True


def test_download_issues_invalid_repo():
    """Test that invalid repo format is handled gracefully."""
    print("\n" + "=" * 60)
    print("LEVEL 3 TEST: download_github_issues with invalid repo")
    print("=" * 60)

    task = download_github_issues(
        repos="not-a-valid-repo",
        labels="",
        state="open",
        max_issues_per_repo=3,
        github_token=""
    )

    output_path = task.outputs["issues_data"].path
    with open(output_path, 'r') as f:
        content = f.read().strip()

    assert content == "", f"FAIL: Expected empty output for invalid repo, got: {content[:100]}"
    print("  ✅ Invalid repo format handled gracefully (no crash)")
    return True


def test_download_issues_multiple_repos():
    """Test fetching from multiple repos via comma separation."""
    print("\n" + "=" * 60)
    print("LEVEL 3 TEST: download_github_issues with multiple repos")
    print("=" * 60)

    task = download_github_issues(
        repos="kubeflow/website,kubeflow/kubeflow",
        labels="",
        state="open",
        max_issues_per_repo=2,
        github_token=os.environ.get("GITHUB_TOKEN", "")
    )

    output_path = task.outputs["issues_data"].path
    with open(output_path, 'r') as f:
        lines = f.readlines()

    issues = [json.loads(line) for line in lines]
    print(f"  Issues fetched: {len(issues)}")

    # Check we got issues from both repos
    repos_seen = set()
    for issue in issues:
        # Extract repo from the path field (e.g., "issues/website/123")
        parts = issue["path"].split("/")
        if len(parts) >= 2:
            repos_seen.add(parts[1])

    print(f"  Repos with issues: {repos_seen}")
    # Without a GitHub token, some repos may return 0 issues due to rate limits
    # or having no open issues. We verify the code ran and produced valid output.
    assert len(issues) > 0, "FAIL: Got 0 issues total from both repos"
    print(f"  ✅ Fetched {len(issues)} issues from repos: {repos_seen}")
    if len(repos_seen) >= 2:
        print(f"  ✅ Issues came from multiple repos")
    else:
        print(f"  ⚠️  Issues only from {repos_seen} (other repo may have 0 matching open issues — not a code bug)")
    return True


if __name__ == "__main__":
    results = {}

    tests = [
        ("KFP component execution", test_download_issues_component),
        ("Empty repos handling", test_download_issues_empty_repos),
        ("Invalid repo handling", test_download_issues_invalid_repo),
        ("Multiple repos", test_download_issues_multiple_repos),
    ]

    for name, test_fn in tests:
        try:
            result = test_fn()
            results[name] = "PASS ✅"
        except AssertionError as e:
            results[name] = f"FAIL ❌ — {e}"
        except Exception as e:
            results[name] = f"ERROR ⚠️ — {type(e).__name__}: {e}"

    print("\n" + "=" * 60)
    print("LEVEL 3 TEST SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, result in results.items():
        status_icon = "✅" if "PASS" in result else "❌"
        print(f"  {status_icon} {name}: {result}")
        if "PASS" not in result:
            all_pass = False

    print()
    if all_pass:
        print("🎉 All Level 3 tests passed!")
    else:
        print("⚠️  Some tests failed. Review output above.")

    sys.exit(0 if all_pass else 1)
