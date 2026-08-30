#!/usr/bin/env bash
set -euo pipefail

secret_dir="${HOME}/.config/sigmetrics-2027"
secret_file="${secret_dir}/openclaw-gaia.env"

install -d -m 700 "${secret_dir}"
chmod 700 "${secret_dir}"
umask 077

read -r -s -p "BRAVE_API_KEY (input hidden): " brave_api_key
printf '\n'

if [[ -z "${brave_api_key}" ]]; then
  echo "No key entered; existing credential was not changed." >&2
  exit 1
fi

temporary_file="$(mktemp "${secret_dir}/.openclaw-gaia.env.XXXXXX")"
trap 'rm -f -- "${temporary_file}"' EXIT
printf 'BRAVE_API_KEY=%q\n' "${brave_api_key}" > "${temporary_file}"
chmod 600 "${temporary_file}"
mv -f -- "${temporary_file}" "${secret_file}"
trap - EXIT
unset brave_api_key

echo "Brave Search credential stored with mode 0600."
