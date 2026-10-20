(function(root){
  'use strict';
  let md,esc,sourceLink;
  const layoutKey='cv-practicum-reading-v1';
  let prefs={left:false,right:false};
  try{const p=JSON.parse(localStorage.getItem(layoutKey));if(p&&typeof p.left==='boolean'&&typeof p.right==='boolean')prefs=p;}catch(e){}
  function configure(options){({md,esc,sourceLink}=options);}
  function toolbar(){return `<div class="reading-toolbar" role="group" aria-label="Панели и ширина теории"><span>Область чтения</span><div class="actions"><button class="btn secondary small reading-left-control" type="button" data-reading="left" aria-controls="sidebar">Темы</button><button class="btn secondary small" type="button" data-reading="right" aria-controls="theory-side">Боковая панель</button><button class="btn small" type="button" data-reading="full">Развернуть теорию</button></div></div>`;}
  function applyLayout(reading){
    document.body.classList.toggle('reading-page',reading);
    document.body.classList.toggle('reading-left-hidden',reading&&prefs.left);
    document.body.classList.toggle('reading-right-hidden',reading&&prefs.right);
    if(!reading)return;
    for(const side of ['left','right']){
      const b=document.querySelector(`[data-reading="${side}"]`);
      if(b){b.setAttribute('aria-expanded',String(!prefs[side]));b.textContent=(prefs[side]?'Показать ':'Скрыть ')+(side==='left'?'темы':'боковую панель');}
    }
    const full=document.querySelector('[data-reading="full"]'),hidden=prefs.left&&prefs.right;
    if(full){full.textContent=hidden?'Вернуть панели':'Развернуть теорию';full.setAttribute('aria-pressed',String(hidden));}
  }
  function changeLayout(kind){
    if(kind==='full'){const hide=!(prefs.left&&prefs.right);prefs={left:hide,right:hide};}
    else prefs[kind]=!prefs[kind];
    if(prefs.left){document.querySelector('#sidebar').classList.remove('open');document.querySelector('#menu-toggle').setAttribute('aria-expanded','false');}
    try{localStorage.setItem(layoutKey,JSON.stringify(prefs));}catch(e){}
    applyLayout(true);
  }
  function showMenu(){
    if(document.body.classList.contains('reading-page')&&prefs.left){prefs.left=false;try{localStorage.setItem(layoutKey,JSON.stringify(prefs));}catch(e){}applyLayout(true);}
  }
  function walkthrough(topic){
    const w=topic.walkthrough;if(!w)return '';
    return `<section class="worked-walkthrough" aria-label="Пошаговый пример"><h2>Разберём на конкретном примере</h2><h3>${esc(w.title)}</h3><div class="walkthrough-given"><strong>Что дано</strong>${md(w.given)}</div><ol class="walkthrough-steps">${w.steps.map(s=>`<li><h4>${esc(s.title)}</h4>${md(s.text)}</li>`).join('')}</ol><div class="callout"><strong>Что важно понять</strong>${esc(w.conclusion)}</div>${demo(topic.id)}<p class="inline-note">Числа в этом разборе подобраны для объяснения механизма из лекции; если не указано иное, это учебные размеры.</p>${sourceLink(w.source)}</section>`;
  }
  function demo(topic){
    if(topic==='vit')return `<section class="study-demo" data-study-demo="patch"><h3>Потрогайте: пиксели → patch embedding</h3><p>RGB-картинка 8×8, патч 2×2. Выберите один из 16 патчей. Проекция в этом демонстраторе задана вручную для наглядности; у настоящего ViT она обучается.</p><div class="study-demo-columns"><div><h4>64 пикселя · рамка выбранного патча</h4><div class="pixel-grid" aria-hidden="true"></div></div><div><h4>16 патчей · нажмите на номер</h4><div class="token-grid" role="group" aria-label="Выбрать патч изображения"></div></div></div><div class="demo-result" aria-live="polite"></div></section>`;
    if(topic==='swin')return `<section class="study-demo" data-study-demo="window"><h3>Потрогайте: кого видит токен до и после сдвига</h3><p>Карта 4×4 токена, M=2. Каждый квадрат — токен с вектором признаков, а не один исходный RGB-пиксель. Нажмите на токен, затем переключите разбиение.</p><label class="demo-switch"><input type="checkbox" data-shift> Сдвинутое разбиение, s=1</label><div class="window-grid" role="group" aria-label="Выбрать query-токен"></div><div class="demo-result" aria-live="polite"></div><p class="inline-note">Показаны разрешённые ключи одного query. Обычное разбиение имеет 4 окна по 4 токена: 4×4×4=64 scores на голову. Реализация shifted windows вычисляет такие же батчи cyclic-окон, а затем маскирует ложные связи завёрнутых краёв. У физической границы разрешённых соседей может оказаться меньше четырёх.</p></section>`;
    return '';
  }
  const pixel=(r,c)=>[r/7,c/7,(r+c)/14];
  function patchData(index){
    const row=Math.floor(index/4)*2,col=(index%4)*2;
    const pixels=[pixel(row,col),pixel(row,col+1),pixel(row+1,col),pixel(row+1,col+1)];
    const mean=k=>pixels.reduce((a,p)=>a+p[k],0)/4;
    return {row,col,flat:pixels.flat(),embedding:[mean(0),mean(1),mean(2),pixels[3][0]-pixels[0][0],pixels[3][1]-pixels[0][1],pixels[3][2]-pixels[0][2]]};
  }
  const number=n=>Number(n.toFixed(3)).toString();
  function drawPatch(section,index=0){
    const p=patchData(index);
    section.dataset.selected=String(index);
    section.querySelector('.pixel-grid').innerHTML=Array.from({length:64},(_,i)=>{
      const r=Math.floor(i/8),c=i%8,selected=r>=p.row&&r<p.row+2&&c>=p.col&&c<p.col+2;
      return `<span class="${selected?'pixel-selected':''}" style="background:rgb(${pixel(r,c).map(x=>Math.round(x*255)).join(',')})"></span>`;
    }).join('');
    section.querySelector('.token-grid').innerHTML=Array.from({length:16},(_,i)=>`<button type="button" data-patch="${i}" aria-pressed="${i===index}" aria-label="Патч ${i+1}, строка ${Math.floor(i/4)+1}, столбец ${i%4+1}">${i+1}</button>`).join('');
    section.querySelector('.demo-result').innerHTML=`<h4>Патч ${index+1}: строки ${p.row}–${p.row+1}, столбцы ${p.col}–${p.col+1} (отсчёт с 0)</h4><p>Четыре RGB-тройки, построчно: сначала верхние два пикселя, затем нижние. Каналы нормированы в [0,1].</p><div class="demo-vector">[${p.flat.map(number).join(', ')}]</div><p>Flatten: 2×2×3 → 12 чисел, их значения пока не изменились. Теперь одна общая Linear(12,6), bias=0:</p><div class="demo-vector">[${p.embedding.map(number).join(', ')}]</div><p>Первые три выхода — средние R,G,B, последние три — разности каналов нижнего правого и верхнего левого пикселя. Например, первый выход = (R₁+R₂+R₃+R₄)/4. Все шесть — линейные комбинации входа: это конкретная матрица 12×6. У обученного ViT значения её весов и смысл координат заранее не фиксированы.</p><p>Для всей картинки: [1,3,8,8] → [1,16,12] → [1,16,6]. До добавления CLS и attention разных патчей здесь ещё не смешивали.</p>`;
  }
  function windowKeys(index,shift){
    const r=Math.floor(index/4),c=index%4,s=shift?1:0;
    return Array.from({length:16},(_,i)=>i).filter(i=>Math.floor((Math.floor(i/4)-s)/2)===Math.floor((r-s)/2)&&Math.floor((i%4-s)/2)===Math.floor((c-s)/2));
  }
  function drawWindow(section,index=5){
    const shift=section.querySelector('[data-shift]').checked,keys=windowKeys(index,shift);
    section.dataset.selected=String(index);
    section.querySelector('.window-grid').innerHTML=Array.from({length:16},(_,i)=>{
      const r=Math.floor(i/4),c=i%4,region=Math.floor(r/2)*2+Math.floor(c/2);
      return `<button type="button" class="region-${region}${keys.includes(i)?' allowed-key':''}${i===index?' query-token':''}" data-window-token="${i}" aria-pressed="${i===index}" aria-label="Токен (${r},${c})${i===index?', выбранный query':''}${keys.includes(i)?', доступный key':''}"><span>(${r},${c})</span><small>${i===index?'QUERY':keys.includes(i)?'KEY':'—'}</small></button>`;
    }).join('');
    const oldGroups=new Set(keys.map(i=>Math.floor(Math.floor(i/4)/2)*2+Math.floor((i%4)/2))).size;
    section.querySelector('.demo-result').innerHTML=`<p><strong>${shift?'SW-MSA, сдвинутое разбиение':'W-MSA, обычное разбиение'}.</strong> Query (${Math.floor(index/4)},${index%4}) читает ${keys.length} ${keys.length===1?'токен':'токена'}: ${keys.map(i=>'('+Math.floor(i/4)+','+i%4+')').join(', ')}. Сам query тоже входит в keys.</p><p>Цветная полоса указывает <em>старое обычное окно</em>, яркая рамка — разрешённый key. В выбранном наборе ${oldGroups} ${oldGroups===1?'старое окно':'старых окна'}. ${shift?'Новые соседи могут уже хранить контекст своих прежних окон, поэтому связь распространяется дальше. На противоположный край карты перехода нет.':'Без смены разбиения этот attention-подслой не связывает разные старые окна.'}</p><p>M=2 — сторона окна: M²=2×2=4 токена. Ширина C вектора каждого из этих токенов — отдельное число. Полная матрица окна имеет M²×M²=4×4=16 scores на голову; после маски не каждый score обязан остаться разрешённым.</p>`;
  }
  function bind(container,reading){
    applyLayout(reading&&Boolean(container.querySelector('.reading-toolbar')));
    for(const b of container.querySelectorAll('[data-reading]'))b.addEventListener('click',()=>changeLayout(b.dataset.reading));
    for(const section of container.querySelectorAll('[data-study-demo]')){
      if(section.dataset.studyDemo==='patch'){
        drawPatch(section);
        section.addEventListener('click',e=>{const b=e.target.closest('[data-patch]');if(!b)return;const i=Number(b.dataset.patch);drawPatch(section,i);section.querySelector(`[data-patch="${i}"]`).focus({preventScroll:true});});
      }else{
        drawWindow(section);
        section.querySelector('[data-shift]').addEventListener('change',()=>drawWindow(section,Number(section.dataset.selected)));
        section.addEventListener('click',e=>{const b=e.target.closest('[data-window-token]');if(!b)return;const i=Number(b.dataset.windowToken);drawWindow(section,i);section.querySelector(`[data-window-token="${i}"]`).focus({preventScroll:true});});
      }
    }
  }
  root.CVStudy={configure,walkthrough,toolbar,bind,showMenu,patchData,windowKeys};
})(typeof window==='undefined'?globalThis:window);
