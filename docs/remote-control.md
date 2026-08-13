# Connecting to the control panel

The engine serves its control panel two ways: plain HTTP for the machine
it runs on, and HTTPS for phones — because iOS only unlocks motion
sensors on secure pages.

| | URL | Who |
|---|---|---|
| Desktop | `http://127.0.0.1:3030` | the machine running the app — no token needed |
| Phone / other devices | `https://<lan-ip>:3031/?key=…` | carried by the QR code — token required |

## Desktop (the machine running collide-o-scope)

1. Launch the app. The panel opens in your default browser automatically.
2. If it doesn't, visit `http://127.0.0.1:3030` yourself.

Localhost connections need no token — the machine trusts itself.

## Phone

**One-time setup (first launch on a new machine):** Windows shows two
firewall prompts — one for port 3030 (HTTP) and one for 3031 (HTTPS).
Approve both for **private networks**, or phones will never connect.

Then, each session:

1. Put the phone on the **same Wi-Fi network** as the PC.
2. In the desktop panel, expand the **REMOTE** section (it starts
   collapsed — click its header) to reveal the QR code.
3. Scan it with the phone camera. The URL is
   `https://<your-lan-ip>:3031/?key=XXXXXXXX` — the key is a per-session
   access token; without it, LAN visitors get 403.
4. **Accept the certificate warning** — the panel serves a self-signed
   certificate:
   - iOS Safari: *This Connection Is Not Private* → **Advanced** →
     **Visit Website**
   - Android Chrome: **Advanced** → **Proceed**

   This is one-time: the certificate persists across app restarts, so
   the phone stays trusting until your LAN IP changes (new venue, new
   router), when it regenerates and asks once more.
5. The panel loads in its mobile layout — single column, master effects
   first, touch-sized controls.

After the first authenticated load, a cookie keeps the phone in for the
rest of the session — but the **token rotates every app launch**, so when
you restart collide-o-scope, re-scan the fresh QR (an old bookmark will
answer 403).

## The library, from any device

- **Add clips**: the **+** beside the Library title uploads videos from the
  device you're holding — a phone's camera roll included. Files stream to
  the engine's `videos/` folder (created automatically on first launch)
  and thumbnail in moments. Name collisions get numbered suffixes.
- **Remove clips**: hover (or tap) a library tile and press its **×** —
  the file moves to the OS **Recycle Bin**, never hard-deleted, so a
  mid-set mistake is recoverable. A clip currently loaded in a layer
  refuses removal until the layer is removed first.

## Performing from the phone

- **XY PAD** — drag anywhere on the square; the position streams at
  ~30 Hz and *holds where you release*, like hardware. Route `Pad X` /
  `Pad Y` to any parameter in the MOD MATRIX.
- **GYRO** — flip the **Stream** toggle; iOS asks for motion permission
  (grant it). Tilt now streams as `Gyro Yaw` / `Pitch` / `Roll` sources.
  This is the feature that *requires* the HTTPS URL — over plain HTTP,
  iOS never offers the sensors.
- Everything else — layers, effects, VHS, LFOs, tap tempo, routings —
  works identically to the desktop panel. All connected devices stay in
  sync through the same state broadcast; the last hand to move a control
  wins.

## Troubleshooting

| Symptom | Cause and cure |
|---|---|
| **403 Access denied** | Stale token from a previous session — re-scan the current QR. |
| **Phone can't reach the panel at all** | Different network, unapproved firewall prompt, or a router with *client isolation* (common on guest Wi-Fi) — use the main network, or share the PC's own hotspot. |
| **Certificate warning returns** | The LAN IP changed, so the certificate regenerated — accept once more. |
| **GYRO shows "sensor needs HTTPS on iOS"** | The page was opened over `http://` — use the QR's `https://…:3031` URL. |
| **GYRO shows "no sensor data"** | The Stream toggle was flipped on a desktop browser — enable it on the phone instead. |
| **QR encodes 127.0.0.1** | No LAN detected at launch (machine offline) — connect to the network and restart the app. |
