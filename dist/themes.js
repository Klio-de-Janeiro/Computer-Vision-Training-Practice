/* Keep theme preferences independent of study records and apply before first paint. */
(function () {
  'use strict';
  const KEY='cv-practicum-theme';
  const themes=new Set(['classic','neon-light','neon-dark']);
  function applyTheme(value, persist=false) {
    const theme=themes.has(value)?value:'neon-dark';
    document.documentElement.dataset.theme=theme;
    document.documentElement.style.colorScheme=theme==='neon-dark'?'dark':'light';
    const meta=document.querySelector('meta[name="theme-color"]');
    if(meta)meta.content=theme==='neon-dark'?'#0a0d1b':theme==='neon-light'?'#f2f4ff':'#202923';
    const picker=document.getElementById('theme-select');
    if(picker)picker.value=theme;
    if(persist)try{localStorage.setItem(KEY,theme);}catch(_){}
  }
  let saved;
  try{saved=localStorage.getItem(KEY);}catch(_){}
  applyTheme(saved);
  document.addEventListener('DOMContentLoaded',()=>{
    const picker=document.getElementById('theme-select');
    if(!picker)return;
    picker.value=document.documentElement.dataset.theme;
    picker.addEventListener('change',()=>applyTheme(picker.value,true));
  });
  window.addEventListener('storage',e=>{if(e.key===KEY)applyTheme(e.newValue);});
})();
