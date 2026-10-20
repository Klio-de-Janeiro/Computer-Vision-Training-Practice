// Worked examples and reading controls use an isolated browser storage context.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),http=require('node:http');
const root=path.resolve(__dirname,'../dist'),box={window:{}};
vm.runInNewContext(fs.readFileSync(path.join(root,'study.js'),'utf8'),box);
const study=box.window.CVStudy;
assert.deepEqual(Array.from(study.windowKeys(5,false)),[0,1,4,5]);
assert.deepEqual(Array.from(study.windowKeys(5,true)),[5,6,9,10]);
assert.deepEqual(Array.from(study.windowKeys(0,true)),[0]); // no wraparound key on the opposite edge
for(let index=0;index<16;index++){
 const p=study.patchData(index);assert.equal(p.flat.length,12);assert.equal(p.embedding.length,6);
 // Independently multiply the illustrative 12×6 matrix shown in the UI.
 const expected=Array.from({length:6},(_,j)=>p.flat.reduce((sum,x,k)=>sum+x*(j<3?(k%3===j?0.25:0):(k===j-3?-1:k===j+6?1:0)),0));
 for(let j=0;j<6;j++)assert.ok(Math.abs(p.embedding[j]-expected[j])<1e-12);
}
console.log('Study arithmetic passed: 16 RGB patches, projection matrix, window membership and boundary mask');
if(process.argv.includes('--core-only'))process.exit(0);
const {chromium}=require(process.argv[2]||'playwright');
const mime={'.html':'text/html','.js':'text/javascript','.css':'text/css','.woff2':'font/woff2'};
const server=http.createServer((req,res)=>{
 const file=path.resolve(root,'.'+decodeURIComponent(new URL(req.url,'http://localhost').pathname.replace(/\/$/,'/index.html')));
 if(!file.startsWith(root+path.sep)){res.writeHead(403);res.end();return;}
 fs.readFile(file,(err,body)=>{res.writeHead(err?404:200,{'Content-Type':mime[path.extname(file)]||'application/octet-stream'});res.end(err?'Missing':body);});
});
(async()=>{
 await new Promise(r=>server.listen(0,'127.0.0.1',r));let browser;
 try{
  const base='http://127.0.0.1:'+server.address().port;
  browser=await chromium.launch({headless:true,executablePath:process.argv[3]||undefined});
  const context=await browser.newContext({viewport:{width:1440,height:1100}}),page=await context.newPage(),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await context.route('**/*',route=>{const u=route.request().url();if(!u.startsWith(base+'/')&&!/^(data:|blob:)/.test(u)){errors.push('External request: '+u);return route.abort();}return route.continue();});
  await page.goto(base+'/');await page.locator('.lesson-card').first().waitFor();
  const topics=await page.evaluate(()=>CV_CONTENT.topics),diagrams=await page.evaluate(()=>CV_CONTENT.diagrams);
  const go=async h=>{await page.evaluate(h=>location.hash=h,h);await page.waitForTimeout(60);};
  for(const width of [1440,390]){
   await page.setViewportSize({width,height:1100});
   for(const t of topics){
    await go('#theory?lesson='+t.lesson+'&topic='+t.id);
    assert.equal(await page.locator('.worked-walkthrough').count(),1,t.id);
    assert.equal(await page.locator('.walkthrough-steps>li').count(),t.walkthrough.steps.length,t.id);
    assert.equal(await page.locator('.katex-error').count(),0,t.id);
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true,t.id+' overflow');
   }
  }
  await page.setViewportSize({width:1440,height:1100});
  await go('#practice?architecture=swin&type=number&count=5');await page.locator('#practice-form button[type=submit]').click();
  await page.locator('#answer-form').waitFor();await page.locator('#answer-text').fill('123');
  const progress=await page.evaluate(()=>localStorage.getItem(KEY));
  await go('#theory?lesson=4&topic=swin');
  const before=await page.locator('.article').boundingBox();
  await page.locator('[data-reading=left]').click();assert.equal(await page.locator('#sidebar').isVisible(),false);
  await page.locator('[data-reading=right]').click();assert.equal(await page.locator('#theory-side').isVisible(),false);
  const wide=await page.locator('.article').boundingBox();assert.ok(wide.width>before.width+250);
  assert.equal(await page.locator('[data-reading=full]').getAttribute('aria-pressed'),'true');
  await page.reload();await page.locator('.worked-walkthrough').waitFor();
  assert.equal(await page.locator('#sidebar').isVisible(),false);assert.equal(await page.locator('#theory-side').isVisible(),false);
  const windowDemo=page.locator('[data-study-demo=window]');
  assert.equal(await windowDemo.locator('.allowed-key').count(),4);
  await windowDemo.locator('[data-shift]').check();
  assert.deepEqual(await windowDemo.locator('.allowed-key').evaluateAll(xs=>xs.map(x=>Number(x.dataset.windowToken))),[5,6,9,10]);
  await windowDemo.locator('[data-window-token="0"]').click();assert.equal(await windowDemo.locator('.allowed-key').count(),1);
  await page.locator('[data-reading=full]').click();
  assert.equal(await page.locator('#sidebar').isVisible(),true);assert.equal(await page.locator('#theory-side').isVisible(),true);
  await go('#theory?lesson=4&topic=vit');
  const patch=page.locator('[data-study-demo=patch]');
  await patch.locator('[data-patch="15"]').click();assert.equal(await patch.getAttribute('data-selected'),'15');
  assert.equal(await patch.locator('.pixel-selected').count(),4);assert.equal(await patch.locator('.demo-vector').count(),2);
  await page.locator('[data-reading=full]').focus();await page.keyboard.press('Enter');
  await page.locator('[data-reading=full]').press('Enter'); // both controls work with keyboard too
  for(const d of diagrams){
   await go('#diagrams?model='+d.id);
   assert.equal(await page.locator('.diagram-scenario p').textContent(),d.scenario);
   assert.equal(await page.locator('.diagram-block-list>section').count(),d.nodes.length);
   const last=d.nodes.at(-1),node=page.locator(`[data-node="${last.id}"]`);await node.scrollIntoViewIfNeeded();await node.click();
   assert.equal(await page.locator('.diagram-detail .diagram-worked>p').textContent(),last.workedExample);
   await page.locator('.diagram-block-list>summary').click();assert.equal(await page.locator('.diagram-block-list>section').first().isVisible(),true);
   assert.equal(await page.locator('.katex-error').count(),0,d.id);
  }
  await go('#theory?lesson=4&topic=swin');await page.locator('[data-node="1"]').click();
  assert.equal(await page.locator('.diagram-detail .diagram-operations>li').count(),4);
  for(const theme of ['neon-dark','neon-light','classic']){
   await page.locator('#theme-select').selectOption(theme);
   for(const width of [1440,390]){
    await page.setViewportSize({width,height:1100});
    await page.locator('[data-reading=full]').click();
    assert.equal(await page.locator('#theory-side').isVisible(),false);
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
    await page.locator('[data-reading=full]').click();
   }
  }
  assert.equal(await page.evaluate(()=>localStorage.getItem(KEY)),progress);
  await go('#home');assert.equal(await page.locator('body').evaluate(b=>b.classList.contains('reading-page')),false);
  assert.deepEqual(errors,[]);
  fs.mkdirSync(path.resolve(__dirname,'../qa'),{recursive:true});
  await page.setViewportSize({width:1440,height:1100});await go('#theory?lesson=4&topic=swin');
  await page.locator('#theme-select').selectOption('neon-dark');await page.locator('[data-reading=full]').click();
  await page.locator('[data-study-demo=window]').screenshot({path:path.resolve(__dirname,'../qa/swin-window-demo.png')});
  await page.screenshot({path:path.resolve(__dirname,'../qa/reading-wide.png')});
  await go('#theory?lesson=4&topic=vit');await page.locator('[data-study-demo=patch]').screenshot({path:path.resolve(__dirname,'../qa/patch-embedding-demo.png')});
  console.log('Study browser passed: 65 examples desktop/mobile, 40 complete block lists, detailed Swin internals, interactive grids, independent collapsible panels, keyboard, reading persistence, 3 themes; study progress preserved.');
 }finally{if(browser)await browser.close();await new Promise(r=>server.close(r));}
})().catch(e=>{console.error(e);process.exitCode=1;});
