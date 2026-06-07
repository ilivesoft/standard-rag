/* @ds-bundle: {"format":3,"namespace":"ILiveDesignSystem_a58218","components":[],"sourceHashes":{"ui_kits/web/app.jsx":"1e6c9d33f571","ui_kits/web/cards.jsx":"ed7f8db85a2d","ui_kits/web/chrome.jsx":"b56578072c4e","ui_kits/web/data.jsx":"f762b95e335e","ui_kits/web/primitives.jsx":"d5cb0f58ca08","ui_kits/web/watch.jsx":"b992a56dc08c"},"inlinedExternals":[],"unexposedExports":[]} */

(() => {

const __ds_ns = (window.ILiveDesignSystem_a58218 = window.ILiveDesignSystem_a58218 || {});

const __ds_scope = {};

(__ds_ns.__errors = __ds_ns.__errors || []);

// ui_kits/web/app.jsx
try { (() => {
// iLive UI Kit — app shell
const {
  useState: useStateApp
} = React;
function HomePage({
  activeCat,
  onCat,
  onOpen
}) {
  let vids = VIDEOS;
  if (activeCat === '지금 라이브') vids = VIDEOS.filter(v => v.live);else if (activeCat === '시청 중') vids = VIDEOS.filter(v => v.progress != null);else if (activeCat === '최근 업로드') vids = VIDEOS.filter(v => !v.live);else if (!['전체'].includes(activeCat)) vids = VIDEOS.filter(v => v.cat === activeCat);
  return /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement(FilterBar, {
    active: activeCat,
    onPick: onCat
  }), vids.length ? /*#__PURE__*/React.createElement(VideoGrid, {
    videos: vids,
    onOpen: onOpen
  }) : /*#__PURE__*/React.createElement("div", {
    style: {
      padding: 80,
      textAlign: 'center',
      color: 'var(--il-text-hint)',
      fontSize: 14
    }
  }, "\uC544\uC9C1 \uC5EC\uAE30\uC5D0 \uD45C\uC2DC\uD560 \uD56D\uBAA9\uC774 \uC5C6\uC5B4\uC694"));
}
function App() {
  const [navOpen, setNavOpen] = useStateApp(true);
  const [route, setRoute] = useStateApp('home'); // 'home' | 'watch'
  const [current, setCurrent] = useStateApp(null);
  const [cat, setCat] = useStateApp('전체');
  const open = v => {
    setCurrent(v);
    setRoute('watch');
    window.scrollTo(0, 0);
  };
  const goHome = () => {
    setRoute('home');
    window.scrollTo(0, 0);
  };
  const watch = route === 'watch';
  return /*#__PURE__*/React.createElement("div", {
    className: "il-root",
    style: {
      minHeight: '100vh'
    }
  }, /*#__PURE__*/React.createElement(TopBar, {
    onMenu: () => setNavOpen(o => !o),
    onLogo: goHome
  }), /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex'
    }
  }, /*#__PURE__*/React.createElement(Sidebar, {
    open: watch ? false : navOpen,
    current: route,
    onNav: goHome
  }), /*#__PURE__*/React.createElement("main", {
    style: {
      flex: 1,
      minWidth: 0
    }
  }, watch ? /*#__PURE__*/React.createElement(WatchPage, {
    video: current,
    onOpen: open
  }) : /*#__PURE__*/React.createElement(HomePage, {
    activeCat: cat,
    onCat: setCat,
    onOpen: open
  }))));
}
ReactDOM.createRoot(document.getElementById('root')).render(/*#__PURE__*/React.createElement(App, null));
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/web/app.jsx", error: String((e && e.message) || e) }); }

// ui_kits/web/cards.jsx
try { (() => {
// iLive UI Kit — cards & grid
function VideoCard({
  video,
  onOpen
}) {
  const ch = CHANNELS[video.ch];
  return /*#__PURE__*/React.createElement("article", {
    className: "il-video-card",
    onClick: () => onOpen(video)
  }, /*#__PURE__*/React.createElement(Thumb, {
    video: video
  }), /*#__PURE__*/React.createElement("div", {
    className: "il-card-meta",
    style: {
      marginTop: 10
    }
  }, /*#__PURE__*/React.createElement(Avatar, {
    ch: video.ch,
    size: 36
  }), /*#__PURE__*/React.createElement("div", {
    style: {
      minWidth: 0
    }
  }, /*#__PURE__*/React.createElement("h3", {
    className: "il-card-title"
  }, video.title), /*#__PURE__*/React.createElement("p", {
    className: "il-card-channel"
  }, ch.name), /*#__PURE__*/React.createElement("p", {
    className: "il-card-stats"
  }, video.live ? `${video.viewers} 명 시청 중` : `조회수 ${video.views}회 · ${video.when}`))));
}
function VideoGrid({
  videos,
  onOpen
}) {
  return /*#__PURE__*/React.createElement("div", {
    className: "il-grid",
    style: {
      padding: '8px 24px 40px'
    }
  }, videos.map(v => /*#__PURE__*/React.createElement(VideoCard, {
    key: v.id,
    video: v,
    onOpen: onOpen
  })));
}
Object.assign(window, {
  VideoCard,
  VideoGrid
});
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/web/cards.jsx", error: String((e && e.message) || e) }); }

// ui_kits/web/chrome.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
// iLive UI Kit — chrome (top bar, sidebar, filter bar)
const {
  useState: useStateChrome
} = React;
function TopBar({
  onMenu,
  onLogo
}) {
  return /*#__PURE__*/React.createElement("header", {
    style: {
      position: 'sticky',
      top: 0,
      zIndex: 20,
      height: 56,
      display: 'flex',
      alignItems: 'center',
      gap: 16,
      padding: '0 16px',
      background: 'var(--il-bg-base)'
    }
  }, /*#__PURE__*/React.createElement("button", {
    className: "il-btn-icon",
    onClick: onMenu
  }, /*#__PURE__*/React.createElement(Icon, {
    name: "menu"
  })), /*#__PURE__*/React.createElement("img", {
    src: "../../assets/ilive-logo-en.png",
    alt: "iLive",
    onClick: onLogo,
    style: {
      height: 24,
      cursor: 'pointer'
    }
  }), /*#__PURE__*/React.createElement("div", {
    style: {
      flex: 1,
      display: 'flex',
      justifyContent: 'center',
      maxWidth: 640,
      margin: '0 auto'
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      width: '100%',
      maxWidth: 540
    }
  }, /*#__PURE__*/React.createElement("input", {
    className: "il-search",
    placeholder: "\uCC44\uB110, \uC601\uC0C1 \uAC80\uC0C9",
    style: {
      flex: 1,
      borderRadius: '20px 0 0 20px',
      borderRight: 'none'
    }
  }), /*#__PURE__*/React.createElement("button", {
    style: {
      background: 'var(--il-surface-el)',
      border: '1px solid var(--il-overlay)',
      borderLeft: 'none',
      borderRadius: '0 20px 20px 0',
      padding: '0 20px',
      cursor: 'pointer'
    }
  }, /*#__PURE__*/React.createElement(Icon, {
    name: "search",
    size: 22,
    style: {
      color: 'var(--il-icon)'
    }
  })))), /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      alignItems: 'center',
      gap: 4
    }
  }, /*#__PURE__*/React.createElement("button", {
    className: "il-btn-icon"
  }, /*#__PURE__*/React.createElement(Icon, {
    name: "video_call"
  })), /*#__PURE__*/React.createElement("button", {
    className: "il-btn-icon"
  }, /*#__PURE__*/React.createElement(Icon, {
    name: "notifications"
  })), /*#__PURE__*/React.createElement(Avatar, {
    ch: "mike",
    size: 32
  })));
}
const NAV_MAIN = [{
  icon: 'home',
  label: '홈'
}, {
  icon: 'sensors',
  label: '라이브'
}, {
  icon: 'subscriptions',
  label: '구독'
}, {
  icon: 'video_library',
  label: '보관함'
}];
const NAV_SUB = [{
  icon: 'history',
  label: '시청 기록'
}, {
  icon: 'schedule',
  label: '나중에 볼 동영상'
}, {
  icon: 'thumb_up',
  label: '좋아요 표시한 동영상'
}];
function Sidebar({
  open,
  current,
  onNav
}) {
  if (!open) {
    return /*#__PURE__*/React.createElement("nav", {
      style: {
        width: 72,
        flexShrink: 0,
        paddingTop: 8
      }
    }, NAV_MAIN.map(n => /*#__PURE__*/React.createElement("div", {
      key: n.label,
      onClick: () => onNav('home'),
      style: {
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: 4,
        padding: '14px 0',
        cursor: 'pointer',
        borderRadius: 10,
        margin: '0 4px'
      },
      onMouseEnter: e => e.currentTarget.style.background = 'var(--il-hover)',
      onMouseLeave: e => e.currentTarget.style.background = 'transparent'
    }, /*#__PURE__*/React.createElement(Icon, {
      name: n.icon,
      size: 24,
      style: {
        color: 'var(--il-text-primary)'
      }
    }), /*#__PURE__*/React.createElement("span", {
      style: {
        fontSize: 10,
        color: 'var(--il-text-primary)'
      }
    }, n.label))));
  }
  const Item = ({
    icon,
    label,
    active
  }) => /*#__PURE__*/React.createElement("div", {
    onClick: () => onNav('home'),
    style: {
      display: 'flex',
      alignItems: 'center',
      gap: 24,
      padding: '0 12px',
      height: 40,
      borderRadius: 10,
      cursor: 'pointer',
      margin: '0 8px',
      background: active ? 'var(--il-surface-el)' : 'transparent'
    },
    onMouseEnter: e => {
      if (!active) e.currentTarget.style.background = 'var(--il-hover)';
    },
    onMouseLeave: e => {
      if (!active) e.currentTarget.style.background = 'transparent';
    }
  }, /*#__PURE__*/React.createElement(Icon, {
    name: icon,
    size: 22,
    fill: active,
    style: {
      color: 'var(--il-text-primary)'
    }
  }), /*#__PURE__*/React.createElement("span", {
    style: {
      fontSize: 14,
      fontWeight: active ? 500 : 400,
      color: 'var(--il-text-primary)'
    }
  }, label));
  return /*#__PURE__*/React.createElement("nav", {
    style: {
      width: 232,
      flexShrink: 0,
      paddingTop: 8,
      overflowY: 'auto'
    }
  }, NAV_MAIN.map((n, i) => /*#__PURE__*/React.createElement(Item, _extends({
    key: n.label
  }, n, {
    active: i === 0
  }))), /*#__PURE__*/React.createElement("div", {
    style: {
      height: 1,
      background: 'var(--il-overlay)',
      margin: '12px 16px'
    }
  }), /*#__PURE__*/React.createElement("div", {
    style: {
      padding: '4px 20px',
      fontSize: 14,
      fontWeight: 500,
      color: 'var(--il-text-primary)'
    }
  }, "\uD68C\uC6D0\uB2D8"), NAV_SUB.map(n => /*#__PURE__*/React.createElement(Item, _extends({
    key: n.label
  }, n))), /*#__PURE__*/React.createElement("div", {
    style: {
      height: 1,
      background: 'var(--il-overlay)',
      margin: '12px 16px'
    }
  }), /*#__PURE__*/React.createElement("div", {
    style: {
      padding: '4px 20px',
      fontSize: 14,
      fontWeight: 500,
      color: 'var(--il-text-primary)'
    }
  }, "\uAD6C\uB3C5"), ['mike', 'note', 'cook', 'dev'].map(ch => /*#__PURE__*/React.createElement("div", {
    key: ch,
    onClick: () => onNav('home'),
    style: {
      display: 'flex',
      alignItems: 'center',
      gap: 20,
      padding: '0 12px',
      height: 40,
      borderRadius: 10,
      cursor: 'pointer',
      margin: '0 8px'
    },
    onMouseEnter: e => e.currentTarget.style.background = 'var(--il-hover)',
    onMouseLeave: e => e.currentTarget.style.background = 'transparent'
  }, /*#__PURE__*/React.createElement(Avatar, {
    ch: ch,
    size: 24
  }), /*#__PURE__*/React.createElement("span", {
    style: {
      fontSize: 14,
      color: 'var(--il-text-primary)'
    }
  }, CHANNELS[ch].name))));
}
function FilterBar({
  active,
  onPick
}) {
  return /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'sticky',
      top: 56,
      zIndex: 10,
      display: 'flex',
      gap: 12,
      padding: '12px 24px',
      background: 'var(--il-bg-base)',
      overflowX: 'auto'
    }
  }, CATS.map(c => /*#__PURE__*/React.createElement("button", {
    key: c,
    className: 'il-chip' + (c === active ? ' active' : ''),
    onClick: () => onPick(c)
  }, c)));
}
Object.assign(window, {
  TopBar,
  Sidebar,
  FilterBar
});
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/web/chrome.jsx", error: String((e && e.message) || e) }); }

// ui_kits/web/data.jsx
try { (() => {
// iLive UI Kit — mock content
const CHANNELS = {
  mike: {
    name: 'MadeByMike',
    handle: '@madebymike',
    color: '#214290',
    initial: 'M',
    subs: '128만'
  },
  note: {
    name: '디자인노트',
    handle: '@designnote',
    color: '#2993D1',
    initial: 'D',
    subs: '54.2만'
  },
  cook: {
    name: '집밥요리',
    handle: '@homecook',
    color: '#29B473',
    initial: '집',
    subs: '210만'
  },
  game: {
    name: '게임라운지',
    handle: '@gamelounge',
    color: '#7A3FB0',
    initial: 'G',
    subs: '88.9만'
  },
  music: {
    name: 'LoFi 라디오',
    handle: '@lofiradio',
    color: '#C2410C',
    initial: '♪',
    subs: '32.1만'
  },
  dev: {
    name: '코드캐스트',
    handle: '@codecast',
    color: '#0E7490',
    initial: '</>',
    subs: '76.5만'
  }
};

// gradient stand-ins for thumbnails (no external images)
const G = (a, b) => `linear-gradient(135deg, ${a}, ${b})`;
const VIDEOS = [{
  id: 'v1',
  ch: 'mike',
  live: true,
  viewers: '12.4만',
  title: '실시간 디자인 시스템 워크숍 — 컴포넌트부터 토큰까지',
  cat: '디자인',
  when: '',
  thumb: G('#214290', '#2993D1'),
  glyph: 'sensors'
}, {
  id: 'v2',
  ch: 'cook',
  dur: '12:34',
  views: '210만',
  title: '10분 완성 김치볶음밥, 자취생 필수 레시피',
  cat: '요리',
  when: '3일 전',
  progress: 45,
  thumb: G('#1f3a2e', '#29B473'),
  glyph: 'restaurant'
}, {
  id: 'v3',
  ch: 'game',
  live: true,
  viewers: '3.8만',
  title: '신작 RPG 첫 공략 생방송 — 보스전 도전',
  cat: '게임',
  when: '',
  thumb: G('#3a2360', '#7A3FB0'),
  glyph: 'sports_esports'
}, {
  id: 'v4',
  ch: 'note',
  dur: '18:09',
  views: '54만',
  title: 'AI가 바꾸는 디자인 시스템의 미래',
  cat: '디자인',
  when: '1주 전',
  thumb: G('#0d2b3a', '#2993D1'),
  glyph: 'palette'
}, {
  id: 'v5',
  ch: 'music',
  live: true,
  viewers: '9,210',
  title: 'LoFi 라디오 — 집중과 휴식을 위한 24시간 방송',
  cat: '음악',
  when: '',
  thumb: G('#3a1f12', '#C2410C'),
  glyph: 'graphic_eq'
}, {
  id: 'v6',
  ch: 'dev',
  dur: '42:51',
  views: '76만',
  title: 'React로 만드는 라이브 스트리밍 UI 처음부터 끝까지',
  cat: '개발',
  when: '2일 전',
  progress: 72,
  thumb: G('#0a2a30', '#0E7490'),
  glyph: 'code'
}, {
  id: 'v7',
  ch: 'cook',
  dur: '08:22',
  views: '33만',
  title: '에어프라이어 감자칩, 기름 없이 바삭하게',
  cat: '요리',
  when: '5일 전',
  thumb: G('#2a2410', '#9a7b1f'),
  glyph: 'lunch_dining'
}, {
  id: 'v8',
  ch: 'note',
  dur: '24:17',
  views: '19만',
  title: '다크 모드 색상 설계 — 명도로 레이어 나누기',
  cat: '디자인',
  when: '2주 전',
  thumb: G('#15171b', '#2c2f37'),
  glyph: 'dark_mode'
}, {
  id: 'v9',
  ch: 'game',
  dur: '1:02:40',
  views: '140만',
  title: '인디 게임 추천 TOP 10 — 2026 상반기',
  cat: '게임',
  when: '4일 전',
  progress: 12,
  thumb: G('#2a1340', '#a23fb0'),
  glyph: 'stadia_controller'
}, {
  id: 'v10',
  ch: 'dev',
  live: true,
  viewers: '1.2만',
  title: '타입스크립트 라이브 코딩 — 디자인 토큰 파서 만들기',
  cat: '개발',
  when: '',
  thumb: G('#0a1f2a', '#0E7490'),
  glyph: 'terminal'
}, {
  id: 'v11',
  ch: 'mike',
  dur: '31:55',
  views: '92만',
  title: '피그마에서 코드로 — 핸드오프 실전 가이드',
  cat: '디자인',
  when: '6일 전',
  thumb: G('#1a2c5a', '#2993D1'),
  glyph: 'design_services'
}, {
  id: 'v12',
  ch: 'music',
  dur: '05:48',
  views: '410만',
  title: '비 오는 날 카페 재즈 모음',
  cat: '음악',
  when: '1개월 전',
  thumb: G('#2a1f3a', '#6d4ca0'),
  glyph: 'music_note'
}];
const CATS = ['전체', '지금 라이브', '디자인', '게임', '음악', '요리', '개발', '최근 업로드', '시청 중'];
const COMMENTS = [{
  ch: 'note',
  text: '토큰 네이밍 컨벤션 부분 너무 깔끔하네요 👏 바로 적용해봐야겠어요',
  when: '2시간 전',
  likes: '1.2천'
}, {
  ch: 'dev',
  text: '라이브로 직접 보니까 이해가 훨씬 잘 됩니다. 다음 방송도 기대할게요!',
  when: '5시간 전',
  likes: '318'
}, {
  ch: 'cook',
  text: '디자인 1도 모르는데 끝까지 봤어요 ㅋㅋ 설명을 너무 잘하셔서',
  when: '1일 전',
  likes: '2.4천'
}];
Object.assign(window, {
  CHANNELS,
  VIDEOS,
  CATS,
  COMMENTS,
  G
});
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/web/data.jsx", error: String((e && e.message) || e) }); }

// ui_kits/web/primitives.jsx
try { (() => {
// iLive UI Kit — primitives
const {
  useState
} = React;
function Icon({
  name,
  size = 24,
  fill = false,
  style = {},
  className = ''
}) {
  return /*#__PURE__*/React.createElement("span", {
    className: 'material-symbols-outlined ' + className,
    style: {
      fontSize: size,
      fontVariationSettings: `'FILL' ${fill ? 1 : 0}`,
      lineHeight: 1,
      ...style
    }
  }, name);
}
function Avatar({
  ch,
  size = 36
}) {
  const c = CHANNELS[ch];
  return /*#__PURE__*/React.createElement("div", {
    style: {
      width: size,
      height: size,
      borderRadius: '50%',
      flexShrink: 0,
      background: c.color,
      color: '#fff',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontWeight: 700,
      fontSize: size * 0.4
    }
  }, c.initial);
}
function FollowButton({
  following,
  onToggle
}) {
  return following ? /*#__PURE__*/React.createElement("button", {
    className: "il-btn-following",
    onClick: onToggle
  }, /*#__PURE__*/React.createElement(Icon, {
    name: "check",
    size: 18
  }), " \uD314\uB85C\uC789") : /*#__PURE__*/React.createElement("button", {
    className: "il-btn-follow",
    onClick: onToggle
  }, "\uD314\uB85C\uC6B0");
}
function LiveBadge() {
  return /*#__PURE__*/React.createElement("span", {
    className: "il-badge-live"
  }, "LIVE");
}
function ViewerBadge({
  count
}) {
  return /*#__PURE__*/React.createElement("span", {
    className: "il-badge-viewers"
  }, /*#__PURE__*/React.createElement(Icon, {
    name: "visibility",
    size: 13
  }), " ", count);
}
function PillButton({
  icon,
  children,
  onClick,
  active
}) {
  return /*#__PURE__*/React.createElement("button", {
    className: "il-btn-pill",
    onClick: onClick,
    style: active ? {
      background: 'var(--il-blue-soft)',
      color: 'var(--il-blue)'
    } : {}
  }, icon && /*#__PURE__*/React.createElement(Icon, {
    name: icon,
    size: 20,
    fill: active
  }), " ", children);
}
function Thumb({
  video,
  radius = 'var(--il-radius-thumb)',
  children
}) {
  return /*#__PURE__*/React.createElement("div", {
    className: "il-thumb",
    style: {
      borderRadius: radius,
      marginBottom: 0
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      width: '100%',
      height: '100%',
      background: video.thumb,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center'
    }
  }, /*#__PURE__*/React.createElement(Icon, {
    name: video.glyph,
    size: 46,
    style: {
      color: 'rgba(255,255,255,0.28)'
    }
  })), video.live && /*#__PURE__*/React.createElement(LiveBadge, null), video.live && /*#__PURE__*/React.createElement(ViewerBadge, {
    count: video.viewers
  }), video.dur && /*#__PURE__*/React.createElement("span", {
    className: "il-duration"
  }, video.dur), video.progress != null && /*#__PURE__*/React.createElement("div", {
    className: "il-progress",
    style: {
      width: video.progress + '%'
    }
  }), children);
}
Object.assign(window, {
  Icon,
  Avatar,
  FollowButton,
  LiveBadge,
  ViewerBadge,
  PillButton,
  Thumb
});
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/web/primitives.jsx", error: String((e && e.message) || e) }); }

// ui_kits/web/watch.jsx
try { (() => {
// iLive UI Kit — watch page
const {
  useState: useStateWatch
} = React;
function Player({
  video,
  playing,
  onToggle
}) {
  const [pct, setPct] = useStateWatch(video.progress || 0);
  return /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'relative',
      width: '100%',
      aspectRatio: '16/9',
      borderRadius: 'var(--il-radius-thumb)',
      overflow: 'hidden',
      background: '#000'
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      inset: 0,
      background: video.thumb,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center'
    }
  }, /*#__PURE__*/React.createElement(Icon, {
    name: video.glyph,
    size: 80,
    style: {
      color: 'rgba(255,255,255,0.22)'
    }
  })), /*#__PURE__*/React.createElement("button", {
    onClick: onToggle,
    style: {
      position: 'absolute',
      inset: 0,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      background: 'transparent',
      border: 'none',
      cursor: 'pointer'
    }
  }, !playing && /*#__PURE__*/React.createElement("div", {
    style: {
      width: 68,
      height: 68,
      borderRadius: '50%',
      background: 'rgba(0,0,0,0.6)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center'
    }
  }, /*#__PURE__*/React.createElement(Icon, {
    name: "play_arrow",
    size: 40,
    fill: true,
    style: {
      color: '#fff'
    }
  }))), /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: 0,
      right: 0,
      bottom: 0,
      background: 'linear-gradient(0deg, rgba(0,0,0,0.85), transparent)',
      padding: '24px 14px 8px'
    }
  }, /*#__PURE__*/React.createElement("div", {
    className: "il-scrubber",
    style: {
      marginBottom: 8
    },
    onClick: e => {
      const r = e.currentTarget.getBoundingClientRect();
      setPct((e.clientX - r.left) / r.width * 100);
    }
  }, /*#__PURE__*/React.createElement("div", {
    className: "il-scrubber-fill",
    style: {
      width: (video.live ? 100 : pct) + '%'
    }
  })), /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      alignItems: 'center',
      gap: 14,
      color: '#fff'
    }
  }, /*#__PURE__*/React.createElement(Icon, {
    name: playing ? 'pause' : 'play_arrow',
    size: 26,
    fill: true,
    style: {
      cursor: 'pointer'
    }
  }), /*#__PURE__*/React.createElement(Icon, {
    name: "skip_next",
    size: 26,
    fill: true,
    style: {
      cursor: 'pointer'
    }
  }), /*#__PURE__*/React.createElement(Icon, {
    name: "volume_up",
    size: 26,
    style: {
      cursor: 'pointer'
    }
  }), video.live ? /*#__PURE__*/React.createElement("span", {
    style: {
      display: 'flex',
      alignItems: 'center',
      gap: 6,
      fontSize: 13,
      fontWeight: 500
    }
  }, /*#__PURE__*/React.createElement("span", {
    style: {
      width: 8,
      height: 8,
      borderRadius: '50%',
      background: 'var(--il-live)'
    }
  }), " \uC2E4\uC2DC\uAC04") : /*#__PURE__*/React.createElement("span", {
    style: {
      fontFamily: 'var(--il-font-mono)',
      fontSize: 12
    }
  }, "18:24 / 48:09"), /*#__PURE__*/React.createElement("span", {
    style: {
      flex: 1
    }
  }), /*#__PURE__*/React.createElement(Icon, {
    name: "settings",
    size: 24,
    style: {
      cursor: 'pointer'
    }
  }), /*#__PURE__*/React.createElement(Icon, {
    name: "cast",
    size: 24,
    style: {
      cursor: 'pointer'
    }
  }), /*#__PURE__*/React.createElement(Icon, {
    name: "fullscreen",
    size: 26,
    style: {
      cursor: 'pointer'
    }
  }))));
}
function UpNextCard({
  video,
  onOpen
}) {
  const ch = CHANNELS[video.ch];
  return /*#__PURE__*/React.createElement("article", {
    onClick: () => onOpen(video),
    style: {
      display: 'flex',
      gap: 8,
      cursor: 'pointer'
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      width: 168,
      flexShrink: 0
    }
  }, /*#__PURE__*/React.createElement(Thumb, {
    video: video,
    radius: "8px"
  })), /*#__PURE__*/React.createElement("div", {
    style: {
      minWidth: 0,
      paddingTop: 2
    }
  }, /*#__PURE__*/React.createElement("h4", {
    style: {
      font: '500 14px/1.3 var(--il-font)',
      color: 'var(--il-text-primary)',
      margin: '0 0 5px',
      display: '-webkit-box',
      WebkitLineClamp: 2,
      WebkitBoxOrient: 'vertical',
      overflow: 'hidden'
    }
  }, video.title), /*#__PURE__*/React.createElement("p", {
    style: {
      font: '400 12px/1.4 var(--il-font)',
      color: 'var(--il-text-sec)',
      margin: 0
    }
  }, ch.name), /*#__PURE__*/React.createElement("p", {
    style: {
      font: '400 12px/1.4 var(--il-font)',
      color: 'var(--il-text-sec)',
      margin: 0
    }
  }, video.live ? `${video.viewers} 명 시청 중` : `조회수 ${video.views}회 · ${video.when}`)));
}
function CommentRow({
  c
}) {
  const ch = CHANNELS[c.ch];
  return /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      gap: 16,
      marginBottom: 20
    }
  }, /*#__PURE__*/React.createElement(Avatar, {
    ch: c.ch,
    size: 40
  }), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      gap: 8,
      alignItems: 'baseline',
      marginBottom: 3
    }
  }, /*#__PURE__*/React.createElement("span", {
    style: {
      fontSize: 13,
      fontWeight: 500,
      color: 'var(--il-text-primary)'
    }
  }, ch.handle), /*#__PURE__*/React.createElement("span", {
    style: {
      fontSize: 12,
      color: 'var(--il-text-sec)'
    }
  }, c.when)), /*#__PURE__*/React.createElement("p", {
    style: {
      fontSize: 14,
      color: 'var(--il-text-primary)',
      margin: '0 0 8px',
      lineHeight: 1.4
    }
  }, c.text), /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      alignItems: 'center',
      gap: 16,
      color: 'var(--il-icon)'
    }
  }, /*#__PURE__*/React.createElement("span", {
    style: {
      display: 'flex',
      alignItems: 'center',
      gap: 6,
      fontSize: 12
    }
  }, /*#__PURE__*/React.createElement(Icon, {
    name: "thumb_up",
    size: 18
  }), " ", c.likes), /*#__PURE__*/React.createElement(Icon, {
    name: "thumb_down",
    size: 18
  }), /*#__PURE__*/React.createElement("span", {
    style: {
      fontSize: 12,
      fontWeight: 500,
      color: 'var(--il-text-primary)'
    }
  }, "\uB2F5\uAE00"))));
}
function WatchPage({
  video,
  onOpen
}) {
  const [playing, setPlaying] = useStateWatch(false);
  const [following, setFollowing] = useStateWatch(false);
  const [liked, setLiked] = useStateWatch(false);
  const ch = CHANNELS[video.ch];
  const upNext = VIDEOS.filter(v => v.id !== video.id).slice(0, 6);
  return /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      gap: 24,
      padding: '24px',
      maxWidth: 1600,
      margin: '0 auto'
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      flex: 1,
      minWidth: 0
    }
  }, /*#__PURE__*/React.createElement(Player, {
    video: video,
    playing: playing,
    onToggle: () => setPlaying(p => !p)
  }), /*#__PURE__*/React.createElement("h1", {
    style: {
      font: '700 20px/1.3 var(--il-font)',
      color: 'var(--il-text-primary)',
      margin: '16px 0 12px'
    }
  }, video.title), /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      alignItems: 'center',
      gap: 16,
      flexWrap: 'wrap'
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      alignItems: 'center',
      gap: 12,
      flex: 1,
      minWidth: 240
    }
  }, /*#__PURE__*/React.createElement(Avatar, {
    ch: video.ch,
    size: 40
  }), /*#__PURE__*/React.createElement("div", {
    style: {
      flex: 1
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 15,
      fontWeight: 500,
      color: 'var(--il-text-primary)'
    }
  }, ch.name), /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 12,
      color: 'var(--il-text-sec)'
    }
  }, "\uAD6C\uB3C5\uC790 ", ch.subs, "\uBA85")), /*#__PURE__*/React.createElement(FollowButton, {
    following: following,
    onToggle: () => setFollowing(f => !f)
  })), /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      gap: 8
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      background: 'var(--il-surface-el)',
      borderRadius: 'var(--il-radius-pill)'
    }
  }, /*#__PURE__*/React.createElement("button", {
    className: "il-btn-pill",
    onClick: () => setLiked(l => !l),
    style: {
      background: 'transparent',
      borderRadius: '20px 0 0 20px',
      borderRight: '1px solid var(--il-overlay)'
    }
  }, /*#__PURE__*/React.createElement(Icon, {
    name: "thumb_up",
    size: 20,
    fill: liked,
    style: {
      color: liked ? 'var(--il-blue)' : undefined
    }
  }), " 1.2\uB9CC"), /*#__PURE__*/React.createElement("button", {
    className: "il-btn-pill",
    style: {
      background: 'transparent',
      borderRadius: '0 20px 20px 0'
    }
  }, /*#__PURE__*/React.createElement(Icon, {
    name: "thumb_down",
    size: 20
  }))), /*#__PURE__*/React.createElement(PillButton, {
    icon: "share"
  }, "\uACF5\uC720"), /*#__PURE__*/React.createElement(PillButton, {
    icon: "bookmark_add"
  }, "\uC800\uC7A5"))), /*#__PURE__*/React.createElement("div", {
    style: {
      background: 'var(--il-surface-el)',
      borderRadius: 'var(--il-radius-md)',
      padding: 12,
      marginTop: 16
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 13,
      fontWeight: 500,
      color: 'var(--il-text-primary)',
      marginBottom: 4
    }
  }, video.live ? `${video.viewers} 명 시청 중 · 실시간` : `조회수 ${video.views}회 · ${video.when}`), /*#__PURE__*/React.createElement("p", {
    style: {
      fontSize: 14,
      color: 'var(--il-text-primary)',
      margin: 0,
      lineHeight: 1.5
    }
  }, "\uC624\uB298 \uBC29\uC1A1\uC5D0\uC11C\uB294 \uB514\uC790\uC778 \uD1A0\uD070\uC744 \uCF54\uB4DC\uB85C \uC62E\uAE30\uB294 \uC804 \uACFC\uC815\uC744 \uD568\uAED8 \uC0B4\uD3B4\uBD10\uC694. \uCC44\uD305\uC73C\uB85C \uC9C8\uBB38 \uC8FC\uC2DC\uBA74 \uC2E4\uC2DC\uAC04\uC73C\uB85C \uB2F5\uBCC0\uB4DC\uB9BD\uB2C8\uB2E4. iLive\uC5D0\uC11C \uB354 \uB9CE\uC740 \uB77C\uC774\uBE0C\uB97C \uB9CC\uB098\uBCF4\uC138\uC694.")), /*#__PURE__*/React.createElement("div", {
    style: {
      marginTop: 24
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 16,
      fontWeight: 700,
      color: 'var(--il-text-primary)',
      marginBottom: 20
    }
  }, "\uB313\uAE00 1,284\uAC1C"), /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      gap: 16,
      marginBottom: 28
    }
  }, /*#__PURE__*/React.createElement(Avatar, {
    ch: "mike",
    size: 40
  }), /*#__PURE__*/React.createElement("input", {
    placeholder: "\uB313\uAE00 \uCD94\uAC00...",
    style: {
      flex: 1,
      background: 'transparent',
      border: 'none',
      borderBottom: '1px solid var(--il-overlay)',
      color: 'var(--il-text-primary)',
      fontSize: 14,
      padding: '6px 0',
      outline: 'none'
    }
  })), COMMENTS.map((c, i) => /*#__PURE__*/React.createElement(CommentRow, {
    key: i,
    c: c
  })))), /*#__PURE__*/React.createElement("aside", {
    style: {
      width: 402,
      flexShrink: 0,
      display: 'flex',
      flexDirection: 'column',
      gap: 12
    }
  }, upNext.map(v => /*#__PURE__*/React.createElement(UpNextCard, {
    key: v.id,
    video: v,
    onOpen: onOpen
  }))));
}
Object.assign(window, {
  Player,
  WatchPage,
  UpNextCard,
  CommentRow
});
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/web/watch.jsx", error: String((e && e.message) || e) }); }

})();
