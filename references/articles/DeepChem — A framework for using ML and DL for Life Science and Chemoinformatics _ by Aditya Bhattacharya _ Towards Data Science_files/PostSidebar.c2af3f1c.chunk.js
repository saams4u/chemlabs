(self.webpackChunklite=self.webpackChunklite||[]).push([[7209],{62182:(e,t,n)=>{"use strict";n.r(t),n.d(t,{PostSidebarContent:()=>X,PostSidebar:()=>W,PostSidebar_customStyleSheet:()=>M,PostSidebar_collection:()=>K,PostSidebar_post:()=>Q});var r=n(28655),o=n.n(r),i=n(50008),a=n.n(i),l=n(63038),s=n.n(l),c=n(59713),u=n.n(c),d=n(71439),p=n(67294),m=n(28859),f=n(84783),v=n(22669),x=n(43689),g=n(86156),E=n(50493),h=n(88065),b=n(47713),S=n(99046),y=n(78886),w=n(49925),P=n(34793),_=n(93125),R=n(33819),I=n(34675),C=n(51684),O=n(31001),k=n(78181),B=n(64504),D=n(67995),A=n(27572),j=n(28309),F=n(67297),L=n(89349),N=n(21146),T=n(27952);function H(){var e=o()(["\n  fragment PostSidebar_post on Post {\n    id\n    clapCount\n    collection {\n      ...auroraHooks_publisher\n      ...PostSidebar_collection\n    }\n    creator {\n      bio\n      name\n      ...UserFollowButton_user\n      ...auroraHooks_publisher\n      ...userUrl_user\n      ...PublisherSidebarFollows_user\n      ...SidebarProfilePic_user\n    }\n    isShortform\n    ...BookmarkButton_post\n    ...CollectionFollowButton_post\n    ...MultiVote_post\n    ...ResponsesIconButton_post\n    ...UserFollowButton_post\n    ...TableOfContents_post\n  }\n  ","\n  ","\n  ","\n  ","\n  ","\n  ","\n  ","\n  ","\n  ","\n  ","\n  ","\n  ","\n"]);return H=function(){return e},e}function U(){var e=o()(["\n  fragment PostSidebar_collection on Collection {\n    id\n    description\n    tagline\n    ...CollectionFollowButton_collection\n    ...collectionUrl_collection\n  }\n  ","\n  ","\n"]);return U=function(){return e},e}function q(){var e=o()(["\n  fragment PostSidebar_customStyleSheet on CustomStyleSheet {\n    ...PublisherSidebarFollows_customStyleSheet\n  }\n  ","\n"]);return q=function(){return e},e}var X=function(e){var t,n=e.customStyleSheet,r=e.maxHeight,o=void 0===r?0:r,i=e.post,a=e.showProfilePic,l=e.visible,s=p.useContext(g.f).openSidebar,c=p.useRef(null),u=(0,j.Iq)(),d=(0,F.v9)((function(e){return e.navigation.currentLocation})),m=(0,w.GT)(i.collection||i.creator)?p.createElement(p.Fragment,null,a&&!!i.creator&&p.createElement(P.$,{user:i.creator}),p.createElement(Y,{post:i,customStyleSheet:n,isVisible:l})):i.collection&&p.createElement(z,{post:i,currentLocation:d}),f={maxHeight:"".concat(o,"px"),overflowY:"scroll",scrollbarWidth:"none","-ms-overflow-style":"none","::-webkit-scrollbar":{display:"none"}};return p.createElement(p.Fragment,null,i.isShortform?m:p.createElement(p.Fragment,null,p.createElement("div",{ref:c},o?p.createElement("div",{className:u(f)},m):m,p.createElement(k.xu,{display:"flex",flexDirection:"row",borderTop:"BASE_LIGHTER",paddingTop:"32px",justifyContent:"flex-start"},p.createElement(k.xu,{display:"flex",marginTop:"-7px",marginBottom:"19px",marginLeft:"-3px",marginRight:"27px"},p.createElement(S.S,{post:i,buttonStyle:"SUBTLE_PADDED",susiEntry:"clap_sidebar",hasDialog:!0})),p.createElement(A.cW,{source:{postId:i.id},extendSource:!0},p.createElement(k.xu,{marginBottom:"19px",marginRight:"16px"},p.createElement(R.h,{responsesCount:null===(t=i.postResponses)||void 0===t?void 0:t.count,location:"sidebar",showResponsesSidebar:s||function(){},allowResponses:i.allowResponses,postId:i.id,isLimitedState:i.isLimitedState})),p.createElement(b.o,{post:i,susiEntry:"bookmark_sidebar"})))),p.createElement(E.o5,{post:i,mode:"SIDEBAR",heightRef:c})))},G={wordBreak:"break-word"},V=function(e){var t;return t={opacity:e?1:0,pointerEvents:e?"auto":"none",willChange:"opacity",position:"fixed",width:"188px",left:"50%",transform:"translateX(406px)",top:"calc(".concat(x.Je,"px + 54px + 14px)")},u()(t,(0,L.nk)("no-preference"),{transition:"opacity 200ms"}),u()(t,"@media all and (max-width: 1230px)",{display:"none"}),t},W=p.forwardRef((function(e,t){var n=e.isClearOfBounds,r=e.isOnPage,o=e.customStyleSheet,i=e.post,l=e.extraWide,c=(0,j.Iq)(),u=p.useContext(m.u6).watchedBounds,d=p.useState(r||!1),f=s()(d,2),v=f[0],x=f[1],g=p.useState(0),E=s()(g,2),h=E[0],b=E[1],S=p.useState(n||!1),y=s()(S,2),w=y[0],P=y[1],R=p.useRef(null),I=p.useRef(null),O=function(e,t){var n,r,o,i,a=null===(n=e.get("byline"))||void 0===n||null===(r=n.ref)||void 0===r?void 0:r.current,l=null===(o=e.get("ghost-track"))||void 0===o||null===(i=o.ref)||void 0===i?void 0:i.current,s=l&&(0,N.L6)(l).top+window.scrollY-window.innerHeight||0;if(a&&l){var c=a.offsetTop+a.offsetHeight+10;return c-s}}(u);p.useEffect($(u,x),[u]);var k=p.useCallback((function(){var e,t;if(function(e,t,n,r,o){if(e.current){var i=t.current,a=["image","bgimage","footer","byline","title","header"],l=(0,C.b2)(e,n,a,o),s=!i||(0,C.b2)(t,n,a,o);r(l&&s)}}(R,I,u,P,{threshold:10}),null!=R&&null!==(e=R.current)&&void 0!==e&&e.clientHeight&&null!==(t=window)&&void 0!==t&&t.innerHeight&&O){var n,r,o=window.innerHeight-O-80;h&&(null==R||null===(n=R.current)||void 0===n?void 0:n.clientHeight)<=o?b(0):o<=(null==R||null===(r=R.current)||void 0===r?void 0:r.clientHeight)&&b(Math.max(o,150))}}),[u,P,h]);p.useEffect((function(){k()}),[]);var B="object"===a()(t)?t:null;p.useEffect((0,C.hE)(k,null==B?void 0:B.current),[k,null==B?void 0:B.current]);var D=w&&v;return p.createElement(A.cW,{source:{susiEntry:"post_sidebar",name:"post_sidebar"}},p.createElement(C.HX,{testId:"post-sidebar",isFixed:!0,scrollableRef:t,sidebarRef:R,topOffset:O,visible:D,extraWide:l},p.createElement(X,{customStyleSheet:o,maxHeight:h,post:i,visible:D})),p.createElement("div",{className:c(V(D)),ref:I},p.createElement(_._U,{postId:i.id,isVisible:D})))})),z=function(e){var t=e.post,n=e.currentLocation,r=(0,j.Iq)(),o=(0,D.n)({name:"heading",scale:"XS"}),i=(0,F.v9)((function(e){return e.config.authDomain})),a=t.collection;return a?p.createElement(k.xu,{marginBottom:"32px"},(null==a?void 0:a.name)&&p.createElement(k.rU,{href:(0,T.WGd)(a,n,i)},p.createElement("h2",{className:r([o,G])},a.name)),(a.tagline||a.description)&&p.createElement(k.xu,{paddingTop:"2px",paddingBottom:"20px"},p.createElement(B.F,{scale:"M",clamp:6},a.tagline||a.description)),p.createElement(f.Fp,{buttonSize:"REGULAR",collection:a,post:t,susiEntry:"follow_sidebar"})):null},Y=function(e){var t=e.post,n=e.customStyleSheet,r=e.isVisible,o=(0,I.Hk)().value,i=(0,F.v9)((function(e){return e.config.authDomain})),a=(0,j.Iq)(),l=(0,D.n)({name:"heading",scale:"XS"}),s=t.creator;if(!s||!s.name)return null;var c=p.createElement("h2",{className:a([l,G])},s.name),u=s?p.createElement(k.rU,{href:(0,T.AWr)(s,i)},c):c;return p.createElement(p.Fragment,null,p.createElement(k.xu,{marginBottom:"32px"},p.createElement(k.xu,{paddingBottom:"5px"},u),s.bio&&p.createElement(k.xu,{paddingTop:"2px"},p.createElement(B.F,{scale:"M"},p.createElement(v.P,{wrapLinks:!0},s.bio))),(null==o?void 0:o.id)!==s.id&&p.createElement(k.xu,{paddingTop:"14px"},p.createElement(O.Bv,{buttonSize:"REGULAR",post:t,user:s,susiEntry:"follow_card"}))),p.createElement(y.Lk.Provider,{value:{postId:t.id}},p.createElement(y.eB,{withBottomBorder:!0,publisher:s,customStyleSheet:n,isVisible:r})))},$=function(e,t){return function(){var n=new IntersectionObserver((function(n){var r=e.get("ghost-track");if(r){var o=n.find((function(e){return e.target===r.ref.current}));o&&t(o.isIntersecting)}else t(!1)}));return e.forEach((function(e){e.ref.current&&n.observe(e.ref.current)})),function(){n.disconnect()}}},M=(0,d.Ps)(q(),y.qy),K=(0,d.Ps)(U(),f.Iq,T.nfI),Q=(0,d.Ps)(H(),w.C5,K,O.sj,h.z,f.b3,S.x,R.K,O.S$,E.tA,T.$mN,y.FB,P.G)},93125:(e,t,n)=>{"use strict";n.d(t,{_U:()=>F,Dk:()=>L});var r=n(28655),o=n.n(r),i=n(59713),a=n.n(i),l=n(63038),s=n.n(l),c=n(46829),u=n(71439),d=n(67294),p=n(12291),m=n(8558),f=n(78181),v=n(98024),x=n(86021),g=n(28309),E=n(90038),h=n(27952);function b(){var e=o()(["\n  fragment ReadNextPostCard_post on Post {\n    id\n    title\n    mediumUrl\n    primaryTopic {\n      name\n      slug\n    }\n    collection {\n      id\n      name\n    }\n    previewImage {\n      id\n      alt\n      focusPercentX\n      focusPercentY\n    }\n  }\n"]);return b=function(){return e},e}var S=(0,u.Ps)(b()),y=(0,p.$j)((function(e){return{mediumOwnedAndOperatedCollectionIds:e.config.mediumOwnedAndOperatedCollectionIds,isCustomDomain:e.client.isCustomDomain}}))((function(e){var t=e.isCustomDomain,n=e.mediumOwnedAndOperatedCollectionIds,r=e.post,o=r&&r.previewImage||{},i=o.focusPercentX,a=o.focusPercentY,l=o.id,s=o.alt,c=r.collection||{},u=c.name,p=c.id,b=r.primaryTopic&&r.primaryTopic.name,S=p&&(n.includes(p)?u:b)||"",y=r&&r.title||"",w=(0,h.jVf)(r,t),P=(0,g.Iq)(),_=d.createElement(f.xu,{marginBottom:"4px"},d.createElement(x.Lh,{tag:"span"},d.createElement(v.F,{scale:"S",color:"DARKER"},d.createElement("div",{className:P({whiteSpace:"nowrap",textOverflow:"ellipsis",overflow:"hidden"})},S))));return d.createElement(f.P3,{href:w},d.createElement(f.xu,{display:"flex",flexDirection:"column",justifyContent:"center",marginTop:"20px"},_,d.createElement(f.xu,{display:"flex",flexDirection:"row"},d.createElement(m.UV,{miroId:l||"",alt:s||"",height:50,width:50,freezeGifs:!0,strategy:E._S.Crop,rules:{marginRight:"9px",marginTop:"4px"},focusPercentX:i||50,focusPercentY:a||50}),d.createElement(f.xu,{display:"flex",flexDirection:"column",width:"130px"},d.createElement(v.F,{scale:"S",color:"DARKER",clamp:3},y)))))})),w=n(34675),P=n(3149),_=n(64504),R=n(27599),I=n(27572),C=n(11348),O=n(6522);function k(){var e=o()(["\n  query ReadNextQuery($postId: ID!) {\n    post(id: $postId) {\n      id\n      readNext {\n        ... on ReadNextItem {\n          reason\n          post {\n            ... on Post {\n              ...ReadNextPostCard_post\n            }\n          }\n        }\n      }\n    }\n  }\n  ","\n"]);return k=function(){return e},e}function B(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function D(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?B(Object(n),!0).forEach((function(t){a()(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):B(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}var A={display:"none"},j={":last-of-type":{xs:A,sm:A,md:A,lg:A,xl:{display:"block"}}},F=function(e){var t=e.isVisible,n=void 0===t||t,r=e.postId,o=d.useState(!1),i=s()(o,2),a=i[0],l=i[1],u=(0,w.Hk)().value,p=(0,c.useLazyQuery)(N),m=s()(p,2),v=m[0],x=m[1],E=x.called,h=x.loading,b=x.error,S=x.data,O=(S=void 0===S?{post:void 0}:S).post,k=u&&L(u)&&!((0,C.yd)()||a),B=(0,R.Av)(),A=(0,g.Iq)(),F=(0,I.Lk)(),T=O&&O.readNext;return d.useEffect((function(){h||!b||T&&T.length||B.event("readNextError",{post:O,postId:r,error:b,readNextLength:T&&T.length||0})}),[h]),d.useEffect((function(){n&&T&&(B.event("readNext.viewed",{position:"sidebar"}),T.slice(0,4).map((function(e,t){var n=e.post,r=e.reason;return B.event("post.clientPresented",{source:(0,I.f0)(D(D({},F),{},{index:t,postFeedReason:r||void 0})),location:"post/".concat(n&&n.id)}),!0})))}),[n,T]),!E&&k&&v({variables:{postId:r||""}}),E&&!h&&!b&&O&&T&&T.length&&k?d.createElement(I.cW,{source:{name:"read_next",sectionType:I.bA.READ_NEXT_SIDEBAR},extendSource:!0},d.createElement(f.xu,{width:"188px"},d.createElement(f.xu,{md:{display:"none"},lg:{width:"780px",margin:"0 24px"},position:"relative",backgroundColor:"BACKGROUND",paddingBottom:"24px",paddingTop:"24px",width:"100%"},d.createElement(f.xu,{position:"absolute",right:"0",top:"-4px"},d.createElement(P.P,{onClick:function(){(0,C.Ph)(),l(!0)},size:"SMALL",isPositionAbsolute:!1})),d.createElement(f.xu,{width:"200px"},d.createElement(_.F1,{scale:"XS"},"Your journey starts here.")),d.createElement(f.xu,{display:"flex",flexDirection:"column",justifyContent:"space-between"},O.readNext.slice(0,4).map((function(e,t){var n=e.post;return n?d.createElement("div",{className:A(j),key:t},d.createElement(y,{key:t,post:n})):null})))))):null};function L(e){return e&&e.createdAt+O.pU>Date.now()}var N=(0,u.Ps)(k(),S)}}]);
//# sourceMappingURL=https://stats.medium.build/lite/sourcemaps/PostSidebar.c2af3f1c.chunk.js.map