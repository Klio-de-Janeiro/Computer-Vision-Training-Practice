const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const {pathToFileURL}=require('node:url');
const C=require('../dist/core.js');
const sandbox={window:{}};vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../dist/content.js'),'utf8'),sandbox);
const data=sandbox.window.CV_CONTENT;
const q=id=>data.questions.find(q=>q.topic===id);
assert.equal(C.grade({type:'short',aliases:['размытие']},{text:'РАЗМЫТИЕ!'}).correct,true);
assert.equal(C.grade({type:'short',aliases:['размытие']},{text:'не размытие'}).correct,false);
assert.equal(C.grade({type:'number',value:1/3,tolerance:.002},{text:'1/3'}).correct,true);
assert.equal(C.grade({type:'number',value:.5,tolerance:.001},{text:'0,5'}).correct,true);
assert.equal(C.grade({type:'number',value:0,tolerance:0},{text:''}).correct,false);
assert.equal(C.grade({type:'number',value:0,tolerance:0},{text:'0/0'}).correct,false);
assert.equal(C.grade({type:'multi',correct:[0,1]},{selected:[1,0]}).correct,true);
assert.equal(C.grade({type:'multi',correct:[0,1]},{selected:[0]}).correct,false);
assert.equal(C.grade(q('rope'),{text:'любые слова'}).auto,false);
const mixed=C.select(data.questions,{count:15},{},[]);
assert.deepEqual([1,2,3,4,5].map(i=>mixed.filter(q=>q.lesson===i).length),[3,3,3,3,3]);
assert.equal(C.select(data.questions,{mode:'errors',count:'all'},{[data.questions[0].id]:{score:0}},[]).length,1);
assert.throws(()=>C.validImport({version:1,records:{bad:{score:3}}},new Set(['bad']),new Set()));
console.log('Core checks passed');
for(const a of data.architectures){
  const pool=C.select(data.questions,{architecture:a.id,count:'all'},{},[]);
  assert.ok(pool.length>=30,a.title);
  assert.ok(pool.every(q=>q.architecture===a.id&&q.lesson===a.lesson));
  const saved=pool[0];
  assert.deepEqual(C.select(data.questions,{architecture:a.id,mode:'saved',count:'all'},{},[saved.id]).map(q=>q.id),[saved.id]);
}
console.log('Architecture filters passed: all 11 banks, saved questions, minimum coverage');

async function browserQA(){
  const {chromium}=require(process.argv[2]||'playwright');
  const browser=await chromium.launch({headless:true,downloadsPath:path.join(__dirname,'../qa/downloads'),executablePath:process.argv[3]||undefined});
  const context=await browser.newContext({viewport:{width:1440,height:1100},acceptDownloads:true});
  const page=await context.newPage();
  const errors=[],failed=[],external=[];
  const base=pathToFileURL(path.join(__dirname,'../dist/index.html')).href;
  page.on('pageerror',e=>errors.push(e.message));
  page.on('response',r=>{if(r.status()>=400)failed.push(r.url()+':'+r.status());});
  await context.route('**/*',route=>{
    if(!/^(file:|data:|blob:)/.test(route.request().url())){external.push(route.request().url());return route.abort();}
    return route.continue();
  });
  await page.goto(base);
  await page.locator('.lesson-card').first().waitFor();
  assert.equal(await page.locator('.lesson-card').count(),5);
  await fs.promises.mkdir(path.join(__dirname,'../qa'),{recursive:true});
  await page.screenshot({path:path.join(__dirname,'../qa/desktop.png'),fullPage:true});
  async function go(hash){await page.evaluate(h=>{location.hash=h;},hash);await page.waitForTimeout(50);}
  // Exercise every topic and all source notebooks, including local images and formula parsing.
  for(const topic of data.topics){await go('#theory?topic='+topic.id+'&lesson='+topic.lesson);assert.equal(await page.locator('.article h1').textContent(),topic.title);assert.equal(await page.locator('.katex-error').count(),0,topic.id);}
  await go('#theory?topic=rope&lesson=4');
  await page.screenshot({path:path.join(__dirname,'../qa/rope.png'),fullPage:true});
  const sources=await page.evaluate(()=>CV_SOURCES.map(n=>({path:n.path,cellCount:n.cellCount})));
  for(const source of sources){await go('#source?path='+encodeURIComponent(source.path));await page.locator('.source-cell').first().waitFor();assert.equal(await page.locator('.source-cell').count(),source.cellCount);await page.locator('img').evaluateAll(imgs=>imgs.forEach(i=>i.loading='eager'));await page.waitForTimeout(120);const broken=await page.locator('.source-cell img').evaluateAll(imgs=>imgs.filter(i=>i.complete&&i.naturalWidth===0).map(i=>i.src));assert.deepEqual(broken,[],source.path);}
  // Start each answer format through the real settings and submit actual UI controls.
  await page.evaluate(()=>{localStorage.clear();location.hash='#home';});await page.reload();
  for(const type of ['single','multi','number','short','open']){
    await go('#practice?type='+type+'&count=5');
    await page.locator('#practice-form button[type=submit]').click();
    if(await page.locator('#replace-session').isVisible())await page.locator('#confirm-start').click();
    await page.locator('#answer-form').waitFor();
    const question=await page.evaluate(()=>getQ());
    if(question.options){for(const i of question.correct)await page.locator('input[name=choice][value="'+i+'"]').check();}
    else await page.locator('#answer-text').fill(type==='number'?String(question.value):type==='short'?question.aliases[0]:'Объяснение своими словами: механизм, пример и ограничения.');
    await page.locator('#answer-form button[type=submit]').click();
    await page.locator('.feedback').first().waitFor();
    if(type==='open'){
      assert.equal(await page.locator('[data-next]').isDisabled(),true);
      await page.locator('[data-rate="0.5"]').click();
    }
    const before=await page.evaluate(()=>JSON.stringify(state.session));await page.reload();assert.equal(await page.evaluate(()=>JSON.stringify(state.session)),before);
    assert.equal(await page.locator('[data-next]').isEnabled(),true);
    if(type!=='open')assert.equal(await page.evaluate(()=>draft(getQ()).score),1);
    await page.locator('[data-bookmark]').click();assert.equal(await page.locator('[data-bookmark]').getAttribute('aria-pressed'),'true');
    await page.locator('[data-next]').click();
    await page.locator('[data-skip]').click();
    assert.equal(await page.evaluate(()=>draft(getQ()).score),0);
  }
  // Finish an entire session and inspect result explanation.
  while(!(await page.evaluate(()=>state.session.done))){
    if(await page.locator('[data-skip]').count())await page.locator('[data-skip]').click();
    await page.locator('[data-next]').click();
  }
  await page.locator('.result-score').waitFor();
  assert.equal(await page.locator('.result-score').count(),1);
  await page.locator('details.result-row').first().locator('summary').click();
  assert.equal(await page.locator('details.result-row').first().locator('.feedback').isVisible(),true);
  await go('#search?q=RoPE');assert.ok(await page.locator('.search-hit').count()>0);
  await go('#progress');const download=page.waitForEvent('download');await page.locator('#export-progress').click();const downloaded=await download;await downloaded.saveAs(path.join(__dirname,'../qa/progress-export.json'));
  const backup=JSON.parse(fs.readFileSync(path.join(__dirname,'../qa/progress-export.json'),'utf8'));assert.ok(Object.keys(backup.records).length>=10);
  page.once('dialog',d=>d.accept());await page.locator('#import-progress').setInputFiles(path.join(__dirname,'../qa/progress-export.json'));await page.waitForTimeout(200);assert.equal(await page.evaluate(()=>state.session),null);
  await go('#home');await page.setViewportSize({width:390,height:844});await page.screenshot({path:path.join(__dirname,'../qa/mobile.png'),fullPage:true});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
  await page.locator('#menu-toggle').click();assert.equal(await page.locator('#sidebar').evaluate(e=>e.classList.contains('open')),true);
  await page.locator('#sidebar a[href="#progress"]').click();assert.equal(await page.locator('#sidebar').evaluate(e=>e.classList.contains('open')),false);
  assert.deepEqual(errors,[],'Browser runtime errors');assert.deepEqual(failed,[],'HTTP errors');assert.deepEqual(external,[],'Unexpected network requests');
  console.log('Browser checks passed: 64 topics, 13 notebooks, 5 answer formats, persistence, results, export/import, mobile; no external requests.');
  await browser.close();
}
if(!process.argv.includes('--core-only'))browserQA().catch(e=>{console.error(e);process.exit(1);});
