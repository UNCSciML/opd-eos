#!/usr/bin/env bash

opd_export_ray_address() {
    local address_file="${RAY_TMPDIR:?RAY_TMPDIR must be set}/ray_current_cluster"
    if [ ! -s "$address_file" ]; then
        echo "Ray address file is missing or empty: $address_file" >&2
        return 1
    fi

    RAY_ADDRESS=$(<"$address_file")
    export RAY_ADDRESS
}

opd_cleanup_ray_session() {
    case "${RAY_TMPDIR:-}" in
        /tmp/opd_sampled_token_*) ;;
        *)
            echo "Refusing to clean unexpected RAY_TMPDIR: ${RAY_TMPDIR:-<unset>}" >&2
            return 2
            ;;
    esac

    pkill -TERM -u "$USER" -f -- "$RAY_TMPDIR" >/dev/null 2>&1 || true
}
