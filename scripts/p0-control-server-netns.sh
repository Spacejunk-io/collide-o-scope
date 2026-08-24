#!/usr/bin/env bash
# Privileged packet/reachability acceptance fixture for the P0 control plane.
# Build the debug binary first, then run this script as root (or through sudo).

set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo "p0-control-server-netns.sh must run as root (use sudo)" >&2
  exit 2
fi

for command in ip curl tcpdump strings grep; do
  command -v "${command}" >/dev/null || {
    echo "missing required command: ${command}" >&2
    exit 2
  }
done

binary=${1:-target/debug/collide-o-scope}
binary=$(realpath "${binary}")
[[ -x ${binary} ]] || {
  echo "debug fixture binary is not executable: ${binary}" >&2
  exit 2
}

suffix="${BASHPID}"
server_ns="cos-p0-server-${suffix}"
client_ns="cos-p0-client-${suffix}"
server_link="cos-p0s-${suffix}"
client_link="cos-p0c-${suffix}"
fixture_root=$(mktemp -d "${TMPDIR:-/tmp}/cos-p0-netns.XXXXXX")
fixture_log="${fixture_root}/fixture.log"
capture="${fixture_root}/lan.pcap"
control_fifo="${fixture_root}/control.fifo"
token=0123456789abcdef0123456789abcdef
server_pid=
capture_pid=

cleanup() {
  set +e
  if [[ -n ${server_pid} ]] && kill -0 "${server_pid}" 2>/dev/null; then
    printf '\n' >&9 2>/dev/null
    wait "${server_pid}" 2>/dev/null
  fi
  if [[ -n ${capture_pid} ]] && kill -0 "${capture_pid}" 2>/dev/null; then
    kill -INT "${capture_pid}" 2>/dev/null
    wait "${capture_pid}" 2>/dev/null
  fi
  exec 9>&- 2>/dev/null
  ip netns del "${client_ns}" 2>/dev/null
  ip netns del "${server_ns}" 2>/dev/null
  rm -rf -- "${fixture_root}"
}
trap cleanup EXIT INT TERM

ip netns add "${server_ns}"
ip netns add "${client_ns}"
ip link add "${server_link}" type veth peer name "${client_link}"
ip link set "${server_link}" netns "${server_ns}"
ip link set "${client_link}" netns "${client_ns}"
ip -n "${server_ns}" link set lo up
ip -n "${client_ns}" link set lo up
ip -n "${server_ns}" addr add 192.0.2.1/24 dev "${server_link}"
ip -n "${client_ns}" addr add 192.0.2.2/24 dev "${client_link}"
ip -n "${server_ns}" -6 addr add 2001:db8:547::1/64 dev "${server_link}"
ip -n "${client_ns}" -6 addr add 2001:db8:547::2/64 dev "${client_link}"
ip -n "${server_ns}" link set "${server_link}" up
ip -n "${client_ns}" link set "${client_link}" up
ip -n "${server_ns}" route add default dev "${server_link}"

mkfifo "${control_fifo}"
exec 9<>"${control_fifo}"
ip netns exec "${client_ns}" tcpdump -U -n -s 0 -i "${client_link}" -w "${capture}" >"${fixture_root}/tcpdump.log" 2>&1 &
capture_pid=$!

COLLIDE_O_SCOPE_P0_TEST_TOKEN="${token}" RUST_LOG=info \
  ip netns exec "${server_ns}" "${binary}" \
    --p0-control-server-fixture \
    --port 3030 \
    --identity-dir "${fixture_root}/identity" \
    <"${control_fifo}" >"${fixture_log}" 2>&1 &
server_pid=$!

for _ in $(seq 1 100); do
  grep -q '^P0_FIXTURE_READY ' "${fixture_log}" && break
  kill -0 "${server_pid}" 2>/dev/null || {
    echo "fixture exited before readiness" >&2
    sed -n '1,120p' "${fixture_log}" >&2
    exit 1
  }
  sleep 0.05
done
grep -q '^P0_FIXTURE_READY loopback=true session=' "${fixture_log}" || {
  echo "fixture did not become ready" >&2
  sed -n '1,120p' "${fixture_log}" >&2
  exit 1
}

curl_common=(--noproxy '*' --silent --show-error --max-time 2)

# The engine namespace can reach both separately owned loopback sockets.
ipv4_local=$(ip netns exec "${server_ns}" curl "${curl_common[@]}" \
  --output /dev/null --write-out '%{http_code}' \
  "http://127.0.0.1:3030/missing?key=${token}")
[[ ${ipv4_local} == 404 ]]
ipv6_local=$(ip netns exec "${server_ns}" curl "${curl_common[@]}" \
  --globoff --output /dev/null --write-out '%{http_code}' \
  "http://[::1]:3030/missing?key=${token}")
[[ ${ipv6_local} == 404 ]]

# A second host cannot reach plaintext 3030 through either address family.
if ip netns exec "${client_ns}" curl "${curl_common[@]}" \
  "http://192.0.2.1:3030/" >/dev/null 2>&1; then
  echo "remote IPv4 unexpectedly reached plaintext port 3030" >&2
  exit 1
fi
if ip netns exec "${client_ns}" curl "${curl_common[@]}" --globoff \
  "http://[2001:db8:547::1]:3030/" >/dev/null 2>&1; then
  echo "remote IPv6 unexpectedly reached plaintext port 3030" >&2
  exit 1
fi

# LAN access is HTTPS-only. The first tokenized navigation mints the distinct
# Secure cookie; the cookie authenticates later routes while a bare request is
# still denied.
cookie_jar="${fixture_root}/cookies.txt"
https_code=$(ip netns exec "${client_ns}" curl "${curl_common[@]}" --insecure \
  --cookie-jar "${cookie_jar}" --output /dev/null --write-out '%{http_code}' \
  "https://192.0.2.1:3031/missing?key=${token}")
[[ ${https_code} == 404 ]]
grep -q $'\t/\tTRUE\t0\tcos_lan\t' "${cookie_jar}"
cookie_code=$(ip netns exec "${client_ns}" curl "${curl_common[@]}" --insecure \
  --cookie "${cookie_jar}" --output /dev/null --write-out '%{http_code}' \
  "https://192.0.2.1:3031/missing")
[[ ${cookie_code} == 404 ]]
bare_code=$(ip netns exec "${client_ns}" curl "${curl_common[@]}" --insecure \
  --output /dev/null --write-out '%{http_code}' \
  "https://192.0.2.1:3031/missing")
[[ ${bare_code} == 403 ]]

# Exercise representative secret/action/thumbnail/upload bytes on the wire;
# every one must remain inside TLS in the client-interface capture.
ip netns exec "${client_ns}" curl "${curl_common[@]}" --insecure \
  --cookie "${cookie_jar}" --header 'Origin: https://192.0.2.1:3031' \
  --header 'Content-Type: application/json' \
  --data-binary '{"action":"export"}' \
  "https://192.0.2.1:3031/controller-profile" >/dev/null
ip netns exec "${client_ns}" curl "${curl_common[@]}" --insecure \
  --cookie "${cookie_jar}" \
  "https://192.0.2.1:3031/thumb/P0_THUMBNAIL_MARKER" >/dev/null
ip netns exec "${client_ns}" curl "${curl_common[@]}" --insecure \
  --cookie "${cookie_jar}" --header 'Origin: https://192.0.2.1:3031' \
  --data-binary 'P0_UPLOAD_MARKER' \
  "https://192.0.2.1:3031/upload?name=P0_UPLOAD_MARKER.mp4" >/dev/null

kill -INT "${capture_pid}"
wait "${capture_pid}"
capture_pid=

for forbidden in \
  "${token}" \
  'cos_lan=' \
  '"action":"export"' \
  'P0_THUMBNAIL_MARKER' \
  'P0_UPLOAD_MARKER'; do
  if strings -a "${capture}" | grep -F -- "${forbidden}" >/dev/null; then
    echo "packet capture exposed forbidden plaintext marker: ${forbidden}" >&2
    exit 1
  fi
done

if grep -F -- "${token}" "${fixture_log}" >/dev/null; then
  echo "fixture log exposed the seeded access token" >&2
  exit 1
fi

printf '\n' >&9
wait "${server_pid}"
server_pid=

echo "P0 netns reachability, TLS flow, packet redaction, and retirement passed"
