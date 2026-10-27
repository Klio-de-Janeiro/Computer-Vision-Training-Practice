// New course material: independent arithmetic, routes, local sources and retained study progress.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),http=require('node:http');
const root=path.resolve(__dirname,'../dist'),box={window:{}};
for(const file of ['content.js','sources.js'])vm.runInNewContext(fs.readFileSync(path.join(root,file),'utf8'),box);
const data=box.window.CV_CONTENT,C=require('../dist/core.js'),topics=data.topics.filter(t=>t.lesson===7),cards=data.codeCards.filter(c=>c.lesson===7);
assert.equal(data.lessons.length,7);assert.equal(topics.length,28);assert.equal(cards.length,33);
assert.equal(C.select(data.questions,{lesson:7,count:'all'},{},[]).length,145);
assert.equal(data.diagrams.filter(d=>d.lesson===7).length,7);
assert.deepEqual(Array.from(box.window.CV_SOURCES.filter(s=>s.lesson===7),s=>s.path),['lesson_7/homework.ipynb','lesson_7/lecture.ipynb']);
const expected={
 'ssl-simclr-2':2*8-2,'ssl-byol-2':.9*2+.1*4,'ssl-beit-2':2*8,'ssl-ibot-4':-Math.log(.5),'ssl-dino-1':2*6-2,'ssl-dinov2-1':(224/14)**2,'ssl-dinov3-1':16**2,
 'ssl-dense-1':Math.floor(15/14)*4+Math.floor(31/14),'ssl-probe-1':8*3+3,'metric-calibration-1':-1,'retrieval-recall-1':1/24,'retrieval-recall-2':1/2,
 'retrieval-map-1':(1+2/3)/2,'retrieval-map-2':(.5+1)/2,'retrieval-ndcg-1':1/Math.log2(3),'metric-verification-1':1/1000,'metric-verification-2':90/100,
 'metric-triplet-1':Math.max(.4-.5+.2,0),'metric-mining-1':Math.max(Math.max(.2,.7)-Math.min(.6,1.1)+.2,0),'metric-pk-1':(6-1)*4,'metric-pk-2':4-1,
 'face-normalized-1':10*.8,'face-margins-1':10*(.8-.1),'face-margins-2':Math.cos(Math.PI/3+Math.PI/6),'face-adaface-1':.4*0+.4,
 'search-flat-1':2-2*.75,'search-flat-4':1000*128*4,'search-hnsw-1':new Set([1,3,8].filter(id=>[1,2,3].includes(id))).size/3,
 'search-ivfpq-1':Math.ceil(16*8/8),'search-lsh-1':1-(Math.PI/2)/Math.PI,'metric-protocol-1':6*24,'metric-protocol-2':6*16,'metric-cache-1':(144+96)*384*4};
const numerical=data.questions.filter(q=>q.lesson===7&&q.type==='number');assert.equal(numerical.length,Object.keys(expected).length);
for(const q of numerical)assert.ok(Math.abs(q.value-expected[q.id.slice(3)])<1e-10,q.id);
assert.ok(data.corrections.some(c=>c.lesson===7&&c.topic==='search-flat'));
assert.equal(data.courseUpdate.revision,'5db8762');
console.log('Lesson 7 content passed: 28 topics, 145 questions, 33 cells, 7 diagrams, 33 independent numeric checks.');
if(process.argv.includes('--core-only'))process.exit(0);
const {chromium}=require(process.argv[2]||'playwright');
const mime={'.html':'text/html','.js':'text/javascript','.css':'text/css','.woff2':'font/woff2','.png':'image/png','.jpg':'image/jpeg','.webp':'image/webp','.avif':'image/avif'};
const server=http.createServer((req,res)=>{const file=path.resolve(root,'.'+decodeURIComponent(new URL(req.url,'http://localhost').pathname.replace(/\/$/,'/index.html')));if(!file.startsWith(root+path.sep)){res.writeHead(403);return res.end();}fs.readFile(file,(err,body)=>{res.writeHead(err?404:200,{'Content-Type':mime[path.extname(file)]||'application/octet-stream'});res.end(err?'Missing':body);});});
(async()=>{
 await new Promise(r=>server.listen(0,'127.0.0.1',r));let browser;
 try{
  const base='http://127.0.0.1:'+server.address().port;
  browser=await chromium.launch({headless:true,executablePath:process.argv[3]||undefined});
  const context=await browser.newContext({viewport:{width:1440,height:1000}}),page=await context.newPage(),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await context.route('**/*',r=>{const u=r.request().url();if(!u.startsWith(base+'/')&&!/^(data:|blob:)/.test(u)){errors.push('External request '+u);return r.abort();}return r.continue();});
  await page.goto(base+'/');await page.locator('.lesson-card').first().waitFor();assert.equal(await page.locator('.lesson-card').count(),7);
  const go=async h=>{await page.evaluate(h=>location.hash=h,h);await page.waitForTimeout(55);};
  assert.ok(await page.locator('.course-update a[href="#lesson?lesson=7"]').count());
  for(const width of [1440,390]){
   await page.setViewportSize({width,height:1000});await page.waitForTimeout(350);
   for(const theme of ['neon-dark','neon-light','classic']){
    await page.selectOption('#theme-select',theme);await go('#lesson?lesson=7');
    assert.equal(await page.locator('.topic-list .topic-link').count(),28);
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true,'Lesson '+theme+' '+width);
   }
   for(const t of topics){
    await go('#theory?lesson=7&topic='+t.id);assert.equal(await page.locator('.article h1').textContent(),t.title);
    assert.equal(await page.locator('.katex-error').count(),0,t.id);assert.equal(await page.locator('.code-cell').count(),t.codeCards.length,t.id);
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true,t.id);
   }
   for(const c of cards){await go('#code?card='+c.id);await page.locator('.code-solution>summary').click();assert.ok((await page.locator('.code-answer code').textContent()).includes(c.code));assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true,c.id);}
  }
  // Source bodies are available only in this local test; public Hosting packages metadata alone.
  for(const source of box.window.CV_SOURCES.filter(s=>s.lesson===7)){
   await go('#source?path='+encodeURIComponent(source.path));await page.locator('.source-cell').first().waitFor();
   assert.equal(await page.locator('.source-cell').count(),source.cellCount,source.path);
   await page.locator('.source-cell img').evaluateAll(imgs=>imgs.forEach(img=>img.loading='eager'));await page.waitForTimeout(500);
   assert.deepEqual(await page.locator('.source-cell img').evaluateAll(imgs=>imgs.filter(i=>i.complete&&i.naturalWidth===0).map(i=>i.src)),[]);
  }
  await go('#practice?lesson=7&type=number&count=5');assert.match(await page.locator('#pool-size').textContent(),/33/);
  await page.locator('#practice-form button[type=submit]').click();await page.locator('#answer-form').waitFor();assert.equal(await page.evaluate(()=>getQ().lesson),7);
  const answer=await page.evaluate(()=>String(getQ().value));await page.locator('#answer-text').fill(answer);await page.locator('#answer-form button[type=submit]').click();await page.locator('.feedback').waitFor();assert.equal(await page.evaluate(()=>draft(getQ()).score),1);
  const records=await page.evaluate(()=>JSON.parse(localStorage.getItem('cv-practicum-v1')).records);await page.reload();await page.locator('.feedback').waitFor();assert.deepEqual(await page.evaluate(()=>JSON.parse(localStorage.getItem('cv-practicum-v1')).records),records);
  await go('#diagrams?lesson=7&model=l7-dino');assert.equal(await page.locator('.diagram-node').count(),6);
  await page.locator('[data-node="2"]').click();assert.equal(await page.locator('[data-node="2"]').getAttribute('aria-pressed'),'true');
  fs.mkdirSync(path.resolve(__dirname,'../qa'),{recursive:true});await page.setViewportSize({width:1440,height:1000});await page.waitForTimeout(350);await page.selectOption('#theme-select','neon-dark');await go('#lesson?lesson=7');await page.screenshot({path:path.resolve(__dirname,'../qa/lesson7-desktop.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});await page.waitForTimeout(350);await page.selectOption('#theme-select','neon-light');await go('#theory?lesson=7&topic=retrieval-recall');await page.screenshot({path:path.resolve(__dirname,'../qa/lesson7-mobile.png'),fullPage:true});
  assert.deepEqual(errors,[]);console.log('Lesson 7 UI passed: all topics/cells desktop/mobile, 3 themes, local sources/images, numeric assessment, persistence and diagram selection.');
 }finally{if(browser)await browser.close();await new Promise(r=>server.close(r));}
})().catch(e=>{console.error(e);process.exitCode=1;});
