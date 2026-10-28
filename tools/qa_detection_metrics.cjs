// Render the worked metric definitions, equations and tables in isolated browser storage.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),http=require('node:http');
const root=path.resolve(__dirname,'../dist'),box={window:{}};
vm.runInNewContext(fs.readFileSync(path.join(root,'content.js'),'utf8'),box);
const data=box.window.CV_CONTENT,katex=require('../dist/vendor/katex.min.js');
const target=data.topics.filter(t=>['map','iou','retrieval-map'].includes(t.id));
for(const t of target)for(const field of ['detail','mathNotes']){
 for(const m of t[field].matchAll(/\$\$([\s\S]*?)\$\$|\$([^$\n]+?)\$/g))katex.renderToString(m[1]||m[2],{throwOnError:true,strict:false});
}
assert.equal(data.questions.length,917);assert.equal(data.topics.length,133);
assert.ok(data.topics.find(t=>t.id==='map').detail.includes('253}{303}'));
console.log('Detection notes checked: definitions, formulas, existing IDs and worked protocols.');
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
  await context.route('**/*',r=>{const u=r.request().url();if(!u.startsWith(base+'/')&&!/^(data:|blob:)/.test(u)){errors.push('External '+u);return r.abort();}return r.continue();});
  await page.goto(base+'/');await page.locator('.lesson-card').first().waitFor();
  const go=async h=>{await page.evaluate(h=>location.hash=h,h);await page.waitForTimeout(90);};
  for(const width of [1440,390]){
   await page.setViewportSize({width,height:1000});await page.waitForTimeout(350);
   for(const theme of ['neon-dark','neon-light','classic']){
    await page.selectOption('#theme-select',theme);
    for(const t of target){
     await go('#theory?lesson='+t.lesson+'&topic='+t.id);assert.equal(await page.locator('.article h1').textContent(),t.title);
     assert.equal(await page.locator('.katex-error').count(),0,t.id+' '+theme);
     const fits=await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth);
     assert.equal(fits,true,'Overflow '+t.id+' '+width);
     if(t.id==='map'){
      assert.ok(await page.locator('.article table').count()>=5);
      assert.equal(await page.locator('.walkthrough-steps>li').count(),6);
      assert.ok((await page.locator('.article').textContent()).includes('не обязана монотонно падать'));
     }
    }
   }
  }
  await go('#practice?lesson=5&topic=map&type=open&count=all');await page.locator('#practice-form button[type=submit]').click();await page.locator('#answer-form').waitFor();
  await page.locator('#answer-text').fill('IoU определяет matching; AP агрегирует PR, mAP усредняет AP по классам.');
  await page.locator('#answer-form button[type=submit]').click();await page.locator('.feedback').waitFor();
  await page.locator('.math-review>summary').click();assert.equal(await page.locator('.feedback .walkthrough-steps>li').count(),6);
  await page.locator('[data-rate="1"]').click();const records=await page.evaluate(()=>JSON.parse(localStorage.getItem('cv-practicum-v1')).records);
  await page.reload();await page.locator('.feedback').waitFor();assert.deepEqual(await page.evaluate(()=>JSON.parse(localStorage.getItem('cv-practicum-v1')).records),records);
  fs.mkdirSync(path.resolve(__dirname,'../qa'),{recursive:true});await page.setViewportSize({width:1440,height:1000});await page.waitForTimeout(350);await page.selectOption('#theme-select','neon-dark');await go('#theory?lesson=5&topic=map');
  await page.screenshot({path:path.resolve(__dirname,'../qa/detection-metrics-desktop.png')});
  await page.setViewportSize({width:390,height:844});await page.waitForTimeout(350);await page.selectOption('#theme-select','neon-light');
  await page.locator('.article h3').filter({hasText:'Один сквозной пример'}).scrollIntoViewIfNeeded();
  await page.screenshot({path:path.resolve(__dirname,'../qa/detection-metrics-mobile.png')});
  assert.deepEqual(errors,[]);console.log('Detection notes UI passed: 3 topics, 3 themes, desktop/mobile tables/formulas, feedback and progress persistence.');
 }finally{if(browser)await browser.close();server.closeAllConnections();await new Promise(r=>server.close(r));}
})().catch(e=>{console.error(e);process.exitCode=1;});
