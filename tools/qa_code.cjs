const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path'),http=require('node:http');
const root=path.resolve(__dirname,'../dist'),box={window:{}};vm.runInNewContext(fs.readFileSync(path.join(root,'content.js'),'utf8'),box);
const data=box.window.CV_CONTENT,C=require('../dist/core.js');
assert.ok(data.codeCards.length>=100);assert.equal(new Set(data.codeCards.map(c=>c.id)).size,data.codeCards.length);
for(const c of data.codeCards){
 assert.ok(c.prompt&&c.code&&c.result&&c.explanation&&c.trap&&c.doc.startsWith('https://'));
 assert.ok(c.runtime!=='cpu'||c.test.trim());
 assert.equal(C.select(data.questions,{collection:'code',card:c.id,count:'all'},{},[]).length,1,c.id);
}
assert.equal(C.select(data.questions,{collection:'code',count:'all'},{},[]).length,data.codeCards.length);
assert.ok(C.select(data.questions,{lesson:6,count:'all'},{},[]).length>=50);
for(const t of data.topics)assert.ok(t.codeCards.length,t.id);
console.log('Code bank integrity passed:',data.codeCards.length,'cards; all topics have extra code; 6 lessons selectable.');
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
  const context=await browser.newContext({viewport:{width:1440,height:1000}}),page=await context.newPage(),errors=[];
  await context.addInitScript(()=>Object.defineProperty(navigator,'clipboard',{value:{writeText:async x=>window.copied=x}}));
  page.on('pageerror',e=>errors.push(e.message));
  await context.route('**/*',r=>{const u=r.request().url();if(!u.startsWith(base+'/')&&!/^(data:|blob:)/.test(u)){errors.push('External request '+u);return r.abort();}return r.continue();});
  await page.goto(base+'/');await page.locator('.lesson-card').first().waitFor();assert.equal(await page.locator('.lesson-card').count(),6);
  const go=async h=>{await page.evaluate(h=>location.hash=h,h);await page.waitForTimeout(45);};
  for(const width of [1440,390]){
   await page.setViewportSize({width,height:1000});
   for(const c of data.codeCards){
    await go('#code?card='+c.id);
    assert.equal(await page.locator('.code-task h2').textContent(),c.focus);
    await page.locator('.code-solution>summary').click();
    assert.ok((await page.locator('.code-answer code').textContent()).includes(c.code));
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true,c.id);
   }
  }
  await page.locator('[data-copy-card]').click();assert.ok((await page.evaluate(()=>window.copied)).includes('model.export'));
  await go('#code?library=NumPy&q=reshape');assert.ok(await page.locator('#code-picker option').count()>0);
  await go('#practice?collection=code&card=pt-tril&count=all');
  assert.match(await page.locator('#pool-size').textContent(),/1/);await page.locator('#practice-form button[type=submit]').click();await page.locator('#answer-form').waitFor();
  assert.equal(await page.evaluate(()=>getQ().codeCard),'pt-tril');
  await page.locator('#answer-text').fill('torch.tril(torch.ones(T,T,dtype=torch.bool))');await page.locator('#answer-form button[type=submit]').click();await page.locator('.feedback').waitFor();
  assert.ok(await page.locator('.feedback pre code').count()>0);await page.locator('[data-rate="1"]').click();
  const saved=await page.evaluate(()=>JSON.parse(JSON.stringify(state.records)));await page.reload();await page.locator('.feedback').waitFor();assert.deepEqual(await page.evaluate(()=>JSON.parse(JSON.stringify(state.records))),saved);
  await go('#materials');assert.equal(await page.locator('.panel h2').count(),11);
  await go('#theory?lesson=6&topic=sam-prompts');assert.ok(await page.locator('.code-cell').count()>0);assert.equal(await page.locator('.katex-error').count(),0);
  await page.locator('.code-cell>summary').first().click();assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
  await go('#practice?lesson=6&collection=materials&count=all');assert.ok(await page.locator('#pool-size').textContent());
  fs.mkdirSync(path.resolve(__dirname,'../qa'),{recursive:true});
  await page.setViewportSize({width:1440,height:1000});await go('#code?card=pt-tril');await page.locator('.code-solution>summary').click();
  await page.screenshot({path:path.resolve(__dirname,'../qa/code-practice-desktop.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});await page.waitForTimeout(350);await page.screenshot({path:path.resolve(__dirname,'../qa/code-practice-mobile.png'),fullPage:true});
  assert.deepEqual(errors,[]);
  console.log('Code UI passed: all cards desktop/mobile, filter, copy, exact-card training, assessment persistence, material coverage and supplemental code cells.');
 }finally{if(browser)await browser.close();await new Promise(r=>server.close(r));}
})().catch(e=>{console.error(e);process.exitCode=1;});
