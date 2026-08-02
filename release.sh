#!/usr/bin/env bash
#
# Release helper for STDPipe.
#
#   ./release.sh 0.4.0
#
# Runs the local pre-flight checks, then creates and pushes an annotated
# v<version> tag. The tag push triggers .github/workflows/release.yaml, which
# builds the distributions from a clean checkout and uploads them to PyPI.
# Nothing is ever uploaded from this machine.
#
# The version itself lives in exactly one place - stdpipe/__init__.py - and is
# picked up from there by pyproject.toml. Bump and commit it before running
# this script; the version and the tag are required to agree.
#
# Options:
#   --skip-tests   do not run the unit tests
#   --skip-build   do not do the local trial build
#   --dry-run      run all the checks, but do not tag or push

set -euo pipefail

VERSION_FILE="stdpipe/__init__.py"
RELEASE_BRANCH="master"
REMOTE="origin"

SKIP_TESTS=0
SKIP_BUILD=0
DRY_RUN=0
VERSION=""

red()   { printf '\033[31m%s\033[0m\n' "$*"; }
green() { printf '\033[32m%s\033[0m\n' "$*"; }
bold()  { printf '\033[1m%s\033[0m\n' "$*"; }

ok()   { green "  ok      $*"; }
info() { printf '  %s\n' "$*"; }
die()  { red "  FAILED  $*"; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --skip-tests) SKIP_TESTS=1 ;;
        --skip-build) SKIP_BUILD=1 ;;
        --dry-run)    DRY_RUN=1 ;;
        -h|--help)    sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*)           die "unknown option: $1" ;;
        *)            [ -z "$VERSION" ] || die "unexpected argument: $1"; VERSION="$1" ;;
    esac
    shift
done

[ -n "$VERSION" ] || { red "usage: ./release.sh <version> [--skip-tests] [--skip-build] [--dry-run]"; exit 1; }

TAG="v$VERSION"

bold "Releasing STDPipe $VERSION as $TAG"

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

# We must be at the top of the working tree, so that all the paths below resolve
cd "$(git rev-parse --show-toplevel 2>/dev/null)" || die "not inside a git repository"
[ -f pyproject.toml ] && [ -f "$VERSION_FILE" ] || die "does not look like the STDPipe repository"

# PEP 440-ish: 1.2, 1.2.3, 1.2.3rc1, 1.2.3a1, 1.2.3.post1 ...
echo "$VERSION" | grep -Eq '^[0-9]+\.[0-9]+(\.[0-9]+)*((a|b|rc)[0-9]+)?(\.post[0-9]+)?$' \
    || die "'$VERSION' does not look like a release version"
ok "version string '$VERSION' is well formed"

BRANCH=$(git rev-parse --abbrev-ref HEAD)
[ "$BRANCH" = "$RELEASE_BRANCH" ] || die "on branch '$BRANCH', releases are made from '$RELEASE_BRANCH'"
ok "on branch $RELEASE_BRANCH"

# Only tracked files matter - the working tree routinely holds scratch files,
# and the build has been verified not to pick them up. What must not happen is
# releasing a commit that does not contain the code being built.
git diff --quiet HEAD -- || die "you have uncommitted changes to tracked files (git status)"
ok "no uncommitted changes to tracked files"

info "fetching from $REMOTE ..."
git fetch --quiet "$REMOTE" "$RELEASE_BRANCH" --tags
if [ -n "$(git rev-list "HEAD..$REMOTE/$RELEASE_BRANCH")" ]; then
    die "local $RELEASE_BRANCH is behind $REMOTE/$RELEASE_BRANCH - pull first"
fi
ok "up to date with $REMOTE/$RELEASE_BRANCH"

if git rev-parse -q --verify "refs/tags/$TAG" >/dev/null; then
    die "tag $TAG already exists locally"
fi
if git ls-remote --exit-code --tags "$REMOTE" "refs/tags/$TAG" >/dev/null 2>&1; then
    die "tag $TAG already exists on $REMOTE"
fi
ok "tag $TAG does not exist yet"

# Version in the single source of truth must match what we are about to tag
CURRENT=$(sed -n 's/^__version__ = ["'"'"']\(.*\)["'"'"']/\1/p' "$VERSION_FILE" | head -1)
[ -n "$CURRENT" ] || die "could not read __version__ from $VERSION_FILE"
if [ "$CURRENT" != "$VERSION" ]; then
    red "  FAILED  $VERSION_FILE says '$CURRENT', but you asked to release '$VERSION'"
    info ""
    info "  Bump it and commit first:"
    info "    sed -i '' 's/^__version__ = .*/__version__ = \"$VERSION\"/' $VERSION_FILE"
    info "    git commit -m 'Version $VERSION' $VERSION_FILE"
    exit 1
fi
ok "$VERSION_FILE declares $VERSION"

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

if [ "$SKIP_TESTS" -eq 0 ]; then
    PYTEST_ARGS="-m unit -q -x --no-header -p no:cacheprovider"
    # Roughly 4x faster when pytest-xdist is around (it is in the dev extras)
    if python3 -c "import xdist" 2>/dev/null; then
        PYTEST_ARGS="$PYTEST_ARGS -n auto"
    fi

    info "running unit tests ..."
    # shellcheck disable=SC2086
    if python3 -m pytest $PYTEST_ARGS >/tmp/stdpipe-release-tests.log 2>&1; then
        ok "unit tests pass ($(grep -Eo '[0-9]+ passed' /tmp/stdpipe-release-tests.log | tail -1))"
    else
        tail -30 /tmp/stdpipe-release-tests.log
        die "unit tests failed (full log in /tmp/stdpipe-release-tests.log)"
    fi
else
    info "skipping tests (--skip-tests)"
fi

# ---------------------------------------------------------------------------
# Trial build - catches packaging and metadata errors before the tag is public
# ---------------------------------------------------------------------------

if [ "$SKIP_BUILD" -eq 0 ]; then
    BUILDDIR=$(mktemp -d)
    trap 'rm -rf "$BUILDDIR"' EXIT
    # Logs live next to, never inside, the distribution directory - otherwise
    # they get swept up by the twine check glob below
    DISTDIR="$BUILDDIR/dist"

    info "trial build ..."
    python3 -m build --outdir "$DISTDIR" >"$BUILDDIR/build.log" 2>&1 \
        || { tail -30 "$BUILDDIR/build.log"; die "build failed"; }

    for f in "stdpipe-$VERSION.tar.gz" "stdpipe-$VERSION-py3-none-any.whl"; do
        [ -f "$DISTDIR/$f" ] || die "expected $f, but build produced: $(ls "$DISTDIR")"
    done
    ok "built stdpipe-$VERSION (sdist + wheel)"

    python3 -m twine check "$DISTDIR"/* >"$BUILDDIR/twine.log" 2>&1 \
        || { cat "$BUILDDIR/twine.log"; die "twine check failed"; }
    ok "twine check passed"

    # The artefacts are thrown away on purpose: what gets uploaded is built by
    # CI from the tagged commit, not whatever happens to be on this machine.
else
    info "skipping trial build (--skip-build)"
fi

# ---------------------------------------------------------------------------
# Tag and push
# ---------------------------------------------------------------------------

if [ "$DRY_RUN" -eq 1 ]; then
    bold ""
    bold "Dry run: all checks passed, nothing was tagged or pushed."
    exit 0
fi

bold ""
bold "All checks passed. About to:"
info "  git tag -a $TAG -m 'STDPipe $VERSION'"
info "  git push $REMOTE $RELEASE_BRANCH"
info "  git push $REMOTE $TAG      <- this triggers the upload to PyPI"
bold ""
printf 'Proceed? [y/N] '
read -r reply
case "$reply" in
    [yY]|[yY][eE][sS]) ;;
    *) red "Aborted."; exit 1 ;;
esac

git tag -a "$TAG" -m "STDPipe $VERSION"
git push "$REMOTE" "$RELEASE_BRANCH"
git push "$REMOTE" "$TAG"

bold ""
green "Pushed $TAG."
info "Watch the release workflow at:"
info "  https://github.com/karpov-sv/stdpipe/actions/workflows/release.yaml"
info "It builds from the tagged commit and publishes to PyPI. Once it is green:"
info "  https://pypi.org/project/stdpipe/$VERSION/"
