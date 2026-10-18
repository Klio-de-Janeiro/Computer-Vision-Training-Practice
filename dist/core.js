(function (root) {
  'use strict';
  const normalize = text => String(text).toLowerCase().replace(/ё/g,'е').replace(/[-–—]/g,' ').replace(/[^\p{L}\p{N}\s]/gu,' ').replace(/\s+/g,' ').trim();
  function grade(q, draft) {
    if (q.type === 'open') return { auto:false, correct:null };
    if (q.type === 'single' || q.type === 'multi') {
      const a=[...new Set(draft.selected || [])].sort((x,y)=>x-y), b=[...q.correct].sort((x,y)=>x-y);
      return {auto:true, correct:a.length===b.length && a.every((v,i)=>v===b[i])};
    }
    if (q.type === 'short') return {auto:true, correct:q.aliases.some(s=>normalize(s)===normalize(draft.text || ''))};
    const raw=String(draft.text || '').trim().replace(',','.');
    let n = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i.test(raw) ? Number(raw) : NaN;
    if (/^[+-]?\d+\s*\/\s*[+-]?\d+$/.test(raw)) {
      const [a,b]=raw.split('/').map(Number); n=b ? a/b : NaN;
    }
    return {auto:true, correct:Number.isFinite(n) && Math.abs(n-q.value)<=q.tolerance};
  }
  function shuffle(items, random=Math.random) {
    const a=[...items]; for (let i=a.length-1;i>0;i--) {const j=Math.floor(random()*(i+1)); [a[i],a[j]]=[a[j],a[i]];} return a;
  }
  function select(questions, settings, records, bookmarks, random=Math.random) {
    let pool=questions.filter(q=>(!settings.lesson || q.lesson===Number(settings.lesson)) && (!settings.topic || q.topic===settings.topic) && (!settings.type || q.type===settings.type));
    if (settings.mode==='errors') pool=pool.filter(q=>records[q.id] && records[q.id].score<1);
    if (settings.mode==='saved') pool=pool.filter(q=>bookmarks.includes(q.id));
    if (settings.mode==='due') pool=pool.filter(q=>records[q.id] && records[q.id].due<=Date.now());
    pool=shuffle(pool,random);
    // Interleave lessons so a short mixed ticket covers the course, not just a lucky sample.
    if (!settings.lesson) {
      const groups=[1,2,3,4,5].map(n=>pool.filter(q=>q.lesson===n)); pool=[];
      while(groups.some(g=>g.length)) for (const g of groups) if(g.length) pool.push(g.pop());
    }
    return pool.slice(0,settings.count==='all' ? pool.length : Number(settings.count || 10));
  }
  function validImport(data, ids, topicIds) {
    if (!data || data.version!==1 || typeof data.records!=='object' || !data.records || Array.isArray(data.records)) throw Error('Это не файл прогресса CV Практикум версии 1.');
    const records={};
    for (const [id,r] of Object.entries(data.records)) {
      if (!ids.has(id)) continue;
      if (!r || ![0,.5,1].includes(r.score) || !Number.isInteger(r.attempts) || r.attempts<1 || r.attempts>100000 || !Number.isFinite(r.due) || !Number.isFinite(r.last)) throw Error('В файле некорректные записи прогресса.');
      records[id]={score:r.score,attempts:r.attempts,due:r.due,last:r.last};
    }
    return {version:1,records,bookmarks:Array.isArray(data.bookmarks)?[...new Set(data.bookmarks.filter(id=>ids.has(id)))]:[],read:Array.isArray(data.read)?[...new Set(data.read.filter(id=>topicIds.has(id)))]:[],session:null};
  }
  const api={normalize,grade,shuffle,select,validImport}; root.CVCore=api;
  if(typeof module!=='undefined') module.exports=api;
})(globalThis);
