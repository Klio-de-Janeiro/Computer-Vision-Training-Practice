(function(root){
  'use strict';
  let data,esc,sourceLink,link,renderText;
  const W=180,H=88,DX=218,DY=126,PAD=34;
  function configure(options){({data,esc,sourceLink,link}=options);renderText=options.md||(s=>`<p>${esc(s)}</p>`);}
  function wrap(text,max=23){
    const lines=[];let line='';
    for(const word of text.split(' ')){
      if(line&&(line+' '+word).length>max){lines.push(line);line='';}
      line+=(line?' ':'')+word;
    }
    if(line)lines.push(line);
    return lines;
  }
  function layout(d){
    const minCol=Math.min(...d.nodes.map(n=>n.col)),minRow=Math.min(...d.nodes.map(n=>n.row));
    const nodes=d.nodes.map(n=>({...n,x:(n.col-minCol)*DX+PAD,y:(n.row-minRow)*DY+PAD}));
    const byId=Object.fromEntries(nodes.map(n=>[n.id,n]));
    let width=Math.max(...nodes.map(n=>n.x+W))+PAD;
    const height=Math.max(...nodes.map(n=>n.y+H))+PAD;
    const edges=d.edges.map((e,i)=>{
      const a=byId[e.from],b=byId[e.to];let path,lx,ly;
      if(e.kind==='skip'||(a.col===b.col&&Math.abs(a.row-b.row)>1)){
        const side=Math.max(a.x,b.x)+W+24+(i%2)*12;
        path=`M ${a.x+W} ${a.y+H/2} H ${side} V ${b.y+H/2} H ${b.x+W}`;
        lx=side+6;ly=(a.y+b.y+H)/2;width=Math.max(width,side+106);
      }else if(a.col===b.col){
        const down=b.y>a.y;const sy=down?a.y+H:a.y,ey=down?b.y:b.y+H;
        path=`M ${a.x+W/2} ${sy} V ${ey}`;lx=a.x+W/2+8;ly=(sy+ey)/2;
      }else{
        const right=b.x>a.x,sx=right?a.x+W:a.x,ex=right?b.x:b.x+W;
        const mid=(sx+ex)/2;
        path=`M ${sx} ${a.y+H/2} H ${mid} V ${b.y+H/2} H ${ex}`;
        lx=mid+6;ly=(a.y+b.y+H)/2;
      }
      const textWidth=e.label.length*5.7;
      const candidates=[{x:lx,y:ly-5},{x:lx-textWidth-12,y:ly-5},
        {x:(a.x+b.x+W-textWidth)/2,y:Math.min(a.y,b.y)-9},
        {x:(a.x+b.x+W-textWidth)/2,y:Math.max(a.y,b.y)+H+18}];
      const labelPosition=candidates.find(p=>p.x>=8&&p.x+textWidth<=width-8&&p.y>=12&&p.y<=height-8&&
        !nodes.some(n=>p.x<n.x+W+3&&p.x+textWidth>n.x-3&&p.y-11<n.y+H+3&&p.y+3>n.y-3));
      return {...e,path,labelPosition};
    });
    const groups=d.groups.map(g=>({...g,x:(g.col-minCol)*DX+PAD-14,y:(g.row-minRow)*DY+PAD-23,
      width:(g.cols-1)*DX+W+28,height:(g.rows-1)*DY+H+40}));
    return {nodes,edges,groups,width,height};
  }
  function nodeText(n){
    const title=wrap(n.title,22),hint=wrap(n.hint,28),titleY=title.length>1?23:29;
    const hintY=title.length>1?60:54;
    return `<text class="diagram-node-title" text-anchor="middle">${title.map((s,i)=>`<tspan x="${W/2}" y="${titleY+i*16}">${esc(s)}</tspan>`).join('')}</text><text class="diagram-node-hint" text-anchor="middle">${hint.map((s,i)=>`<tspan x="${W/2}" y="${hintY+i*13}">${esc(s)}</tspan>`).join('')}</text>`;
  }
  function detail(d,n){return `<div class="diagram-detail-heading"><span class="eyebrow">БЛОК ${d.nodes.indexOf(n)+1} / ${d.nodes.length}</span><h4>${esc(n.title)}</h4></div><p>${esc(n.detail)}</p><small>${esc(n.hint)}</small><div class="diagram-worked"><h5>Конкретно в этом примере</h5>${renderText(n.workedExample||'')}${n.operations?`<ol class="diagram-operations">${n.operations.map(s=>`<li><strong>${esc(s.title)}</strong>${renderText(s.text)}</li>`).join('')}</ol>`:''}</div>`;}
  function figure(d,scope){
    const g=layout(d),uid='architecture-'+scope+'-'+d.id;
    const markers=['flow','skip','context','training'].map(kind=>`<marker id="${uid}-${kind}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="diagram-arrow kind-${kind}" d="M 0 0 L 10 5 L 0 10 z"/></marker>`).join('');
    return `<figure class="architecture-figure" data-model="${d.id}"><figcaption><h3>${esc(d.title)}</h3><p>${esc(d.summary)}</p></figcaption><div class="diagram-scenario"><strong>Пример для этой схемы</strong><p>${esc(d.scenario||'')}</p></div><p class="diagram-instruction">Нажмите на блок, чтобы разобрать его роль. <span>Схему можно прокручивать.</span></p><div class="diagram-controls"><button class="btn secondary small" type="button" data-diagram-expand aria-expanded="${scope==='catalog'}">${scope==='catalog'?'Свернуть схему':'Показать всю схему'}</button></div><div class="diagram-viewport${scope==='catalog'?' expanded':''}" tabindex="0" aria-label="Прокручиваемая схема ${esc(d.title)}"><svg class="model-diagram" width="${g.width}" height="${g.height}" viewBox="0 0 ${g.width} ${g.height}" role="group" aria-labelledby="${uid}-title ${uid}-desc"><title id="${uid}-title">${esc(d.title)}</title><desc id="${uid}-desc">${esc(d.summary)} Сплошные стрелки показывают основной поток, пунктир обозначает дополнительные связи. Нажмите Enter на блоке для пояснения.</desc><defs>${markers}</defs>${g.groups.map(group=>`<g class="diagram-repeat"><rect x="${group.x}" y="${group.y}" width="${group.width}" height="${group.height}" rx="16"/><text x="${group.x+10}" y="${group.y+15}">${esc(group.label)}</text></g>`).join('')}${g.edges.map(e=>`<g class="diagram-edge kind-${e.kind}" data-edge="${e.from}:${e.to}"><path d="${e.path}" marker-end="url(#${uid}-${e.kind})"/>${e.label&&e.labelPosition?`<text x="${e.labelPosition.x}" y="${e.labelPosition.y}">${esc(e.label)}</text>`:''}</g>`).join('')}${g.nodes.map((n,i)=>`<g class="diagram-node ${n.kind}${i===0?' selected':''}" transform="translate(${n.x},${n.y})" role="button" tabindex="0" aria-label="${esc(n.title+': '+n.hint)}" aria-pressed="${i===0}" aria-controls="${uid}-detail" data-node="${n.id}"><rect width="${W}" height="${H}" rx="11"/>${nodeText(n)}</g>`).join('')}</svg></div><div class="diagram-legend"><span><i class="kind-flow"></i>Поток данных</span><span><i class="kind-skip"></i>Shortcut / residual</span><span><i class="kind-context"></i>Контекст / признаки</span><span><i class="kind-training"></i>Обучение</span></div><details class="diagram-connections"><summary>Связи между блоками · ${d.edges.length}</summary><ul>${d.edges.map(e=>`<li>${esc(d.nodes.find(n=>n.id===e.from).title)} → ${esc(d.nodes.find(n=>n.id===e.to).title)}${e.label?` <small>(${esc(e.label)})</small>`:''}</li>`).join('')}</ul></details><div class="diagram-detail" id="${uid}-detail" aria-live="polite">${detail(d,d.nodes[0])}</div><details class="diagram-block-list"><summary>Разобрать все блоки по порядку · ${d.nodes.length}</summary>${d.nodes.map(n=>`<section>${detail(d,n)}</section>`).join('')}</details><div class="diagram-source">${sourceLink(d.source)}</div></figure>`;
  }
  function panel(topic){
    const models=(topic.diagrams||[]).map(id=>data.diagrams.find(d=>d.id===id));
    if(!models.length)return '';
    return `<section class="architecture-schemes" aria-label="Схемы архитектур" data-diagram-panel><div class="section-head"><h2>Схема архитектуры</h2><a class="source-link" href="${link('diagrams',{lesson:topic.lesson,model:models[0].id})}">Каталог схем ↗</a></div>${models.length>1?`<label class="diagram-model-label">Выберите схему<select data-diagram-picker>${models.map(d=>`<option value="${d.id}">${esc(d.title)}</option>`).join('')}</select></label>`:''}<div data-diagram-display>${figure(models[0],'article')}</div></section>`;
  }
  function page(p){
    const lesson=Number(p.get('lesson'))||0;
    const models=data.diagrams.filter(d=>!lesson||d.lesson===lesson);
    const chosen=models.find(d=>d.id===p.get('model'))||models[0];
    return `<div class="intro"><div class="eyebrow">ВХОДЫ → БЛОКИ → ВЫХОДЫ</div><h1>Архитектуры моделей.</h1><p>${data.diagrams.length} схем по лекциям: путь признаков, повторяющиеся блоки, ветви и задачи голов.</p></div><div class="diagram-filters"><label>Материал<select id="diagram-lesson"><option value="0">Все уроки со схемами</option>${data.lessons.filter(l=>data.diagrams.some(d=>d.lesson===l.id)).map(l=>`<option value="${l.id}" ${l.id===lesson?'selected':''}>0${l.id} · ${esc(l.short)}</option>`).join('')}</select></label><label>Модель или компонент<select id="diagram-model">${models.map(d=>`<option value="${d.id}" ${d.id===chosen?.id?'selected':''}>${esc(d.title)}</option>`).join('')}</select></label></div>${chosen?`<div class="diagram-catalog-layout"><nav class="diagram-catalog-nav" aria-label="Модели каталога">${models.map(d=>`<a class="${d.id===chosen.id?'active':''}" href="${link('diagrams',{lesson:lesson||'',model:d.id})}"><small>УРОК 0${d.lesson}</small>${esc(d.title)}</a>`).join('')}</nav><section class="panel diagram-catalog-main">${figure(chosen,'catalog')}<div class="actions"><a class="btn secondary small" href="${link('theory',{lesson:chosen.lesson,topic:chosen.topic})}">Теория и формулы</a><a class="btn small" href="${link('practice',{lesson:chosen.lesson,topic:chosen.topic,count:'all'})}">Практика по теме →</a></div></section></div>`:'<p>Схем для этого урока пока нет.</p>'}`;
  }
  function bind(container){
    for(const picker of container.querySelectorAll('[data-diagram-picker]'))picker.addEventListener('change',()=>{
      const display=picker.closest('[data-diagram-panel]').querySelector('[data-diagram-display]');
      display.innerHTML=figure(data.diagrams.find(d=>d.id===picker.value),'article');
    });
    function activate(target){
      const expand=target.closest('[data-diagram-expand]');
      if(expand){
        const viewport=expand.closest('figure').querySelector('.diagram-viewport');
        const expanded=viewport.classList.toggle('expanded');
        expand.setAttribute('aria-expanded',String(expanded));
        expand.textContent=expanded?'Свернуть схему':'Показать всю схему';
        return;
      }
      const node=target.closest('[data-node]');if(!node)return;
      const fig=node.closest('[data-model]'),d=data.diagrams.find(d=>d.id===fig.dataset.model);
      const item=d.nodes.find(n=>n.id===node.dataset.node);
      for(const other of fig.querySelectorAll('[data-node]')){const selected=other===node;other.classList.toggle('selected',selected);other.setAttribute('aria-pressed',String(selected));}
      fig.querySelector('.diagram-detail').innerHTML=detail(d,item);
    }
    for(const section of container.querySelectorAll('.architecture-schemes,.diagram-catalog-main')){
      section.addEventListener('click',e=>activate(e.target));
      section.addEventListener('keydown',e=>{if(['Enter',' '].includes(e.key)&&e.target.matches('[data-node]')){e.preventDefault();activate(e.target);}});
    }
    const lesson=container.querySelector('#diagram-lesson'),model=container.querySelector('#diagram-model');
    lesson?.addEventListener('change',()=>{location.hash=link('diagrams',{lesson:Number(lesson.value)||''});});
    model?.addEventListener('change',()=>{location.hash=link('diagrams',{lesson:Number(lesson.value)||'',model:model.value});});
  }
  root.CVDiagrams={configure,panel,page,bind,layout};
})(window);
