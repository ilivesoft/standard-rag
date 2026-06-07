# iLive (아이라이브) — Design System

A dark-mode design system for **iLive**, a Korean **live-streaming + video-on-demand** platform. iLive's product surface is built around live broadcasts and recorded video: browsing, watching, following creators, and chatting/commenting — all in a single dark theme tuned around the iLive blue brand.

> **How this system came to be.** The foundation was seeded from a YouTube-style dark-mode token spec (the structural conventions of a modern video platform — surfaces, spacing, radii, the card/chip/player vocabulary). It has been **re-skinned to iLive's real brand**: the brand-accent role (CTAs, follow buttons, progress bars, active states) is iLive **blue**, not red. Red is reserved strictly for the universal **LIVE / recording** indicator. The result is original to iLive, not a clone of any one product.

---

## Sources provided

- **Brand logos** (uploaded by the user, copied into `assets/`):
  - `assets/ilive-logo-en.png` — English wordmark "iLive" + arc
  - `assets/ilive-logo-kr.png` — wordmark + Korean "아이라이브" inline
  - `assets/ilive-logo-kr-stack.png` — wordmark + Korean stacked
- **Foundation spec** — a dark-mode video-platform token sheet (colors, type, spacing, radii, component recipes) pasted into the brief. Adapted, not copied verbatim.

No codebase or Figma file was provided. UI kit recreations below are built from the brand spec + logo, not from production source — flag if real source becomes available so screens can be made pixel-exact.

---

## Brand at a glance

| | |
|---|---|
| **Name** | iLive (아이라이브) |
| **Domain** | Live streaming + VOD video platform |
| **Theme** | Dark only |
| **Primary brand color** | iLive blue `#2993D1` |
| **Deep brand color** | Navy `#214290` (logo "i", gradients) |
| **Live accent** | Red `#FF2D2D` — LIVE/recording badges ONLY |
| **Typeface** | Pretendard Variable (self-hosted) → `-apple-system, sans-serif`; Roboto Mono for timecodes |
| **Logo mark** | Italic, bold "iLive" with a hand-drawn arc above; navy "i", blue "Live" |

---

## Index — what's in this folder

| File / folder | Purpose |
|---|---|
| `README.md` | This file — context, content + visual foundations, iconography, index |
| `colors_and_type.css` | All design tokens: color, type scale, spacing, radii, elevation + semantic classes |
| `components.css` | Component recipes (buttons, chips, badges, cards, player, toast, inputs) |
| `SKILL.md` | Agent Skill manifest for reuse in Claude Code |
| `assets/` | Brand logos (and any imagery) |
| `preview/` | Small HTML cards that populate the Design System tab |
| `ui_kits/web/` | iLive web app UI kit — JSX components + interactive `index.html` |

---

## CONTENT FUNDAMENTALS — how iLive writes

iLive is a **bilingual Korean/English** product. UI labels and creator-facing copy are primarily **Korean**; the brand name and a few platform terms stay in English ("iLive", "LIVE", "VOD").

- **Voice:** friendly, concise, creator-positive. Speaks *to* the viewer plainly — no corporate stiffness, no hype.
- **Person:** addresses the user with informal-polite Korean (해요체 / 합니다 mix). English UI uses imperative verbs ("Follow", "Watch live", "Share").
- **Casing (English):** Sentence case for body and labels; ALL-CAPS reserved for the `LIVE` badge only. Never Title Case buttons.
- **Numbers / metadata:** abbreviated Korean counts — `조회수 210만회`, `12만 명 시청 중`, `3일 전`. English equivalents abbreviate too: `2.1M views`, `120K watching`, `3 days ago`. Meta is always joined with a middot ` · `.
- **Live language:** present-tense and immediate — "지금 라이브", "12.4만 명 시청 중", "Live now". VOD uses past/relative time — "3일 전", "1.2M views".
- **Emoji:** **not** used in core platform chrome (nav, buttons, titles). Emoji appear only inside **user-generated content** — live chat messages and comments — never in system copy.
- **Tone examples:**
  - Follow CTA: `팔로우` / `Follow` → after: `팔로잉 ✓` / `Following`
  - Empty state: "아직 시청 기록이 없어요" / "Nothing here yet — start watching to fill this up."
  - Toast: "채널을 팔로우했어요" / "Following — you'll get notified when they go live."
  - Live meta line: `MadeByMike · 12.4만 명 시청 중`

---

## VISUAL FOUNDATIONS

**Overall vibe.** Quiet, dark, content-first. The interface recedes (near-black cool-grey surfaces) so thumbnails and live streams carry all the color. iLive blue is a precise accent, not a wash. The mood is calm and modern — closer to a focused media player than a busy social feed.

**Color.**
- Backgrounds are **cool-shifted neutrals**, not pure grey: base `#0F1013`, surface `#1A1C21`, elevated `#24262C`, borders/chips `#383B43`. The subtle blue undertone harmonizes with the brand.
- **Layering is by brightness, not shadow** — surfaces step up in lightness to read as "closer". Shadows appear only on truly floating layers (menus, modals, toasts).
- **Brand blue `#2993D1`** is used surgically: follow/CTA buttons, the player progress fill, active nav/tab/chip state, focus rings, links, and brand gradients. Never as a page or card fill, never for body text.
- **Navy `#214290`** pairs with blue in the brand gradient (`120deg, navy → blue`) used on hero/banner surfaces and the logo.
- **Red `#FF2D2D`** is semantic-only: the `LIVE` badge and recording dots. It never decorates.

**Typography.** Pretendard Variable throughout — a Korean-first variable sans that pairs clean Hangul with Latin. Weights: 400 (meta/body), 500 (titles, buttons, card titles), 700 (headings/display), 900 reserved for occasional brand display. Tight scale — display 24/700, heading 18/700, card title 14/500, meta 13/400, label 11/600. Letter-spacing only nudged negative on display (-0.3px) and positive on the all-caps LIVE label.

**Spacing & layout.** 4-based scale (4/6/8/12/16/24/32/48/64). Page padding 24px desktop / 16px mobile. Section gap 32px. Card grid: `auto-fill minmax(280px,1fr)`, 12px column gap, 16px row gap, thumbnail-to-meta gap 10px. Responsive: 1 col (<640), 2 col (640–1024), 3–4 col (>1024). The left nav rail is a fixed surface; everything else scrolls.

**Backgrounds & imagery.** No decorative illustration, no texture, no noise. The "imagery" is the content itself — 16:9 thumbnails and live frames with `border-radius:12px`. A **bottom scrim gradient** (`transparent → rgba(0,0,0,.78)`) sits over hero/featured frames so overlaid text and badges stay legible. Imagery color is left untouched (true to source) except a `brightness(0.85)` darken on card hover.

**Corner radii.** Everything is softly rounded but capped: badges 4px, toasts/chips-small 6px, menus 8px, **thumbnails 12px**, pills (buttons/filter chips) 20px, avatars/icon-buttons 50%. Progress bars are square (0px). Nothing exceeds the thumbnail's 12px except full pills/circles.

**Cards.** No border, no drop shadow. A card is just: rounded thumbnail + a row (channel avatar 36px + 2-line clamped title + 2 lines of secondary meta). Elevation comes from the surrounding base being darker, not from the card lifting.

**Borders.** Hairline `1px` in `--il-overlay` (#383B43) for inputs, ghost buttons, dividers, and floating-layer edges. No heavy borders. Progress bars are the only sub-1px-feeling element (3px square fills).

**Shadows / elevation.** Used sparingly: menus `0 4px 24px rgba(0,0,0,.5)`, toasts `0 4px 16px`, modals `0 16px 48px`. Resting UI has none.

**Animation.** Restrained and fast. Transitions ~0.15s ease on color/background/filter. Hover = brightness or background-overlay shift; no bounce, no scale-up on cards. The scrubber grows 3px→5px on hover and reveals a 13px handle. Live dots may pulse subtly. No page-load choreography.

**Hover / press / focus states.**
- Hover (rows, icon buttons): white overlay `rgba(255,255,255,.08)`.
- Active/press: `rgba(255,255,255,.13)` — overlay darkens, no shrink.
- Brand buttons darken on hover (`#2993D1 → #1F7AB0`).
- Thumbnails darken (`brightness .85`) on card hover.
- Focus-visible: `2px` brand-blue outline, `2px` offset.

**Transparency & blur.** Translucency is reserved for **scrims over media** (duration/viewer badges on `rgba(0,0,0,.85)`, hero scrim gradient) and the white hover overlays. No frosted-glass blur in chrome — surfaces are solid so text stays crisp on a dark theme.

---

## ICONOGRAPHY

- **System:** **Material Symbols (Outlined)**, loaded from the Google Fonts CDN. This matches the clean, single-weight outline language expected of a dark video platform and avoids hand-rolled SVGs.
  ```html
  <link rel="stylesheet"
    href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0">
  ```
  Usage: `<span class="material-symbols-outlined">search</span>`. Common glyphs: `search`, `menu`, `home`, `subscriptions`, `video_library`, `notifications`, `account_circle`, `more_vert`, `thumb_up`, `share`, `play_arrow`, `pause`, `volume_up`, `fullscreen`, `cast`, `sensors` (live), `chat`.
- **Sizing:** 20px inline, 24px navigation, 36px player controls. Icon color `--il-icon` (#8C9199) at rest → `--il-text-primary` on hover. Active nav icons may use the **FILL 1** axis to read as selected.
- **Filled vs outlined:** outlined is the default state; the filled variant signals the *active/selected* state (e.g. current nav item, liked).
- **Logo mark:** the iLive wordmark — italic bold "iLive" (navy "i" `#214290`, blue "Live" `#2993D1`) under a hand-drawn blue arc. Use `assets/ilive-logo-en.png` in chrome; the Korean lockups for marketing/footer. Always on dark; keep clear space ≈ the height of the "i".
- **Emoji:** never in chrome; allowed only inside user content (chat/comments).
- **Unicode as icon:** the middot `·` is the canonical metadata separator; `✓` may follow "팔로잉/Following". No other unicode-as-icon usage.

See `SKILL.md` for reuse instructions.
