#!/usr/bin/env bash
# Resolve the release version for ACI wheel builds.
#
# On release/tag builds we want the wheel version to be the tag itself
# (v0.1.2 -> 0.1.2), not setuptools_scm's guessed next development
# version (for example 0.1.3.devN).

set -euo pipefail

log() {
    printf '[prepare_aci_release_version] %s\n' "$*" >&2
}

is_release_tag() {
    local tag="$1"
    [[ "$tag" =~ ^v?[0-9]+([.][0-9]+){1,2}([._+-]?[0-9A-Za-z]+)*$ ]]
}

normalize_ref() {
    local ref="$1"
    ref="${ref#refs/tags/}"
    ref="${ref#refs/heads/}"
    ref="${ref#origin/}"
    printf '%s' "$ref"
}

fetch_tags() {
    if ! git rev-parse --git-dir >/dev/null 2>&1; then
        return
    fi

    if [[ "$(git rev-parse --is-shallow-repository 2>/dev/null || echo false)" == "true" ]]; then
        git fetch --unshallow --tags --force >/dev/null 2>&1 || git fetch --tags --force >/dev/null 2>&1 || true
    else
        git fetch --tags --force >/dev/null 2>&1 || true
    fi
}

exact_release_tag() {
    local tag
    while IFS= read -r tag; do
        if is_release_tag "$tag"; then
            printf '%s' "$tag"
            return 0
        fi
    done < <(git tag --points-at HEAD 2>/dev/null || true)
    return 1
}

env_release_tag() {
    local var value tag
    for var in \
        ACI_COMMIT_TAG \
        CI_COMMIT_TAG \
        GIT_TAG \
        TAG_NAME \
        CODE_TAG \
        RELEASE_TAG \
        CI_COMMIT_REF_NAME \
        GIT_BRANCH \
        BRANCH_NAME
    do
        value="${!var:-}"
        if [[ -z "$value" ]]; then
            continue
        fi

        tag="$(normalize_ref "$value")"
        if is_release_tag "$tag"; then
            log "found release tag from ${var}=${value}"
            printf '%s' "$tag"
            return 0
        fi
    done
    return 1
}

checkout_tag_if_available() {
    local tag="$1"
    local tag_commit head_commit

    if ! git rev-parse -q --verify "refs/tags/${tag}^{commit}" >/dev/null; then
        log "tag ${tag} is not available locally after fetching tags; refusing to publish an unverified release version"
        return 2
    fi

    tag_commit="$(git rev-list -n1 "$tag")"
    head_commit="$(git rev-parse HEAD)"
    if [[ "$head_commit" != "$tag_commit" ]]; then
        log "checking out ${tag} (${tag_commit}) instead of current HEAD ${head_commit}"
        git checkout --detach "$tag" >&2
    fi
}

fetch_tags

if tag="$(exact_release_tag)"; then
    log "HEAD is exactly on release tag ${tag}"
    printf '%s\n' "${tag#v}"
    exit 0
fi

if tag="$(env_release_tag)"; then
    checkout_tag_if_available "$tag"
    printf '%s\n' "${tag#v}"
    exit 0
fi

log "no release tag found; leaving setuptools_scm to derive a development version"
exit 1
