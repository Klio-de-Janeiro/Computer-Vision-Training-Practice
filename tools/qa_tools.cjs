// Tooling content, scoped training and offline desktop/mobile rendering.
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path'),http=require('node:http');
const root=path.resolve(__dirname,'../dist'),box={window:{}};
vm.runInNewContext(fs.readFileSync(path.join(root,'content.js'),'utf8'),box);
const data=box.window.CV_CONTENT,C=require('../dist/core.js'),guides=data.toolGuides,tooling=data.questions.filter(q=>q.collection==='tooling');
assert.equal(guides.length,3);assert.equal(tooling.length,94);
assert.equal(data.topics.filter(t=>t.extension==='tooling').length,18);
assert.equal(data.codeCards.filter(c=>c.runtime==='tooling').length,40);
for(const g of guides){
 assert.equal(g.topics.length,6);assert.ok(g.docs.length>=2&&g.project&&g.prerequisite&&g.checks.length>=3);
 const pool=C.select(data.questions,{collection:'tooling',framework:g.id,count:'all'},{},[]);
 assert.ok(pool.length>=30);assert.ok(pool.every(q=>q.framework===g.id&&q.source.url));
 for(const id of g.topics){const t=data.topics.find(t=>t.id===id);assert.equal(t.framework,g.id);assert.equal(t.lesson,0);assert.ok(t.codeCards.length>=2);}
}
// Independent worked numeric answers, including uneven batches and accumulation remainder.
const values={'lt-module-concept-3':.2,'lt-logging-concept-4':(4*1+1*6)/5,'lt-optimization-concept-10':Math.ceil(10/4),'lt-optimization-concept-11':8*4*2,'lt-checkpoint-concept-14':3*2,'tb-scalars-concept-22':1*10+3,'tb-images-concept-26':-1*.25+.5,'tb-embeddings-concept-31':100,'tb-embeddings-concept-32':1/(1+1),'tb-hparams-concept-34':3*2*4,'mf-metrics-concept-43':Math.min(.6,.2,.4),'mf-artifacts-concept-46':1e6*4/1e6,'mf-comparison-concept-52':(.7+.8+.9)/3};
for(const [id,value] of Object.entries(values))assert.ok(Math.abs(data.questions.find(q=>q.id===id).value-value)<1e-12,id);
console.log('Training tools integrity passed: 18 topics, 94 questions, 40 cells; independent numerical checks.');
if(process.argv.includes('--core-only'))process.exit(0);
const {chromium}=require(process.argv[2]||'playwright');
const server=http.createServer((req,res)=>{const file=path.resolve(root,'.'+decodeURIComponent(new URL(req.url,'http://localhost').pathname.replace(/\/$/,'/index.html')));if(!file.startsWith(root+path.sep)){res.writeHead(403);return res.end();}fs.readFile(file,(err,body)=>{res.writeHead(err?404:200,{'Content-Type':{'.html':'text/html','.js':'text/javascript','.css':'text/css','.woff2':'font/woff2'}[path.extname(file)]||'application/octet-stream'});res.end(err?'Missing':body);});});
(async()=>{
 await new Promise(r=>server.listen(0,'127.0.0.1',r));let browser;
 try{
  const base='http://127.0.0.1:'+server.address().port;
  browser=await chromium.launch({headless:true,executablePath:process.argv[3]||undefined});
  const context=await browser.newContext({viewport:{width:1440,height:1000}}),page=await context.newPage(),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await context.addInitScript(()=>Object.defineProperty(navigator,'clipboard',{value:{writeText:async x=>window.copied=x}}));
  await context.route('**/*',r=>{const u=r.request().url();if(!u.startsWith(base+'/')&&!/^(data:|blob:)/.test(u)){errors.push('External request '+u);return r.abort();}return r.continue();});
  await page.goto(base+'/');await page.locator('.lesson-card').first().waitFor();
  const go=async h=>{await page.evaluate(h=>location.hash=h,h);await page.waitForTimeout(60);};
  for(const width of [1440,390]){
   await page.setViewportSize({width,height:1000});await page.waitForTimeout(350);
   for(const theme of ['neon-dark','neon-light','classic']){
    await page.selectOption('#theme-select',theme);await go('#tools');
    assert.equal(await page.locator('.tool-guide').count(),3);
    await page.locator('.tool-setup summary').first().click();
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true,'Tools '+theme+' @ '+width);
   }
   for(const g of guides){
    await go('#tools?framework='+g.id);assert.equal(await page.locator('.tool-guide').count(),1);
    for(const id of g.topics){
     await go('#theory?topic='+id);assert.equal(await page.locator('.article h1').textContent(),data.topics.find(t=>t.id===id).title);
     assert.ok((await page.locator('.article .back-link').count())===0); // back link is in article navigation
     assert.equal(await page.locator('.article-nav .back-link').getAttribute('href'),'#tools?framework='+g.id);
     assert.equal(await page.locator('.katex-error').count(),0,id);
     assert.ok(await page.locator('.article a.source-link[href^="https://"]').count()>0,id);
     assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true,id);
    }
   }
   for(const c of data.codeCards.filter(c=>c.runtime==='tooling')){
    await go('#code?card='+c.id);await page.locator('.code-solution>summary').click();
    assert.ok((await page.locator('.code-answer code').textContent()).includes(c.code));
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true,c.id);
   }
  }
  await go('#practice?collection=tooling&framework=mlflow&count=all');
  assert.match(await page.locator('#pool-size').textContent(),/30/);
  await page.selectOption('#framework-filter','lightning');assert.match(await page.locator('#pool-size').textContent(),/34/);
  await go('#code?card=lt-resume');await page.locator('.code-solution>summary').click();await page.locator('[data-copy-card]').click();assert.ok((await page.evaluate(()=>window.copied)).includes('ckpt_path="last.ckpt"'));
  await page.locator('a.btn[href^="#practice"]').first().click();await page.locator('#practice-form').waitFor();assert.match(await page.locator('#pool-size').textContent(),/1/);
  await page.locator('#practice-form button[type=submit]').click();await page.locator('#answer-form').waitFor();
  assert.equal(await page.evaluate(()=>getQ().codeCard),'lt-resume');
  assert.equal(await page.locator('.question-setup').count(),1);
  await page.locator('#answer-text').fill('resumed.fit(model, train_dataloaders=loader, ckpt_path="last.ckpt")');
  await page.locator('#answer-form button[type=submit]').click();await page.locator('.feedback').waitFor();await page.locator('[data-rate="1"]').click();
  const records=await page.evaluate(()=>JSON.parse(localStorage.getItem('cv-practicum-v1')).records);
  await page.reload();await page.locator('.feedback').waitFor();assert.deepEqual(await page.evaluate(()=>JSON.parse(localStorage.getItem('cv-practicum-v1')).records),records);
  await go('#lesson?lesson=3');assert.equal(await page.locator('.topic-list .topic-link').count(),16);
  fs.mkdirSync(path.resolve(__dirname,'../qa'),{recursive:true});
  await page.setViewportSize({width:1440,height:1000});await page.waitForTimeout(350);await go('#tools?framework=lightning');
  await page.screenshot({path:path.resolve(__dirname,'../qa/training-tools-desktop.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});await page.waitForTimeout(350);await go('#theory?topic=mf-metrics');
  await page.screenshot({path:path.resolve(__dirname,'../qa/training-tools-mobile.png'),fullPage:true});
  assert.deepEqual(errors,[]);console.log('Training tools UI passed: three themes, desktop/mobile, 18 theories, 40 cards, source links, filters, copy, assessment persistence and course separation.');
 }finally{if(browser)await browser.close();await new Promise(r=>server.close(r));}
})().catch(e=>{console.error(e);process.exitCode=1;});
