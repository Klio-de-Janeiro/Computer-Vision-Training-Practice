const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),http=require('node:http');
const root=path.resolve(__dirname,'../dist'),out=path.resolve(__dirname,'../qa');
const box={window:{}};
vm.runInNewContext(fs.readFileSync(path.join(root,'content.js'),'utf8'),box);
vm.runInNewContext(fs.readFileSync(path.join(root,'diagrams.js'),'utf8'),box);
const data=box.window.CV_CONTENT;
for(const d of data.diagrams){
  const layout=box.window.CVDiagrams.layout(d);
  assert.ok(layout.width>0&&layout.height>0);
  assert.ok(layout.edges.every(e=>!e.path.includes('NaN')));
  for(const n of layout.nodes)assert.ok(n.x>=0&&n.y>=0&&n.x+180<=layout.width&&n.y+88<=layout.height,d.id);
}
console.log('Graph layout passed:',data.diagrams.length,'schemes');
if(process.argv.includes('--layout-only'))process.exit(0);
const {chromium}=require(process.argv[2]||'playwright');
const mime={'.html':'text/html','.js':'text/javascript','.css':'text/css','.woff2':'font/woff2','.woff':'font/woff'};
const server=http.createServer((req,res)=>{
  const file=path.resolve(root,'.'+decodeURIComponent(new URL(req.url,'http://localhost').pathname.replace(/\/$/,'/index.html')));
  if(!file.startsWith(root+path.sep)){res.writeHead(403);res.end();return;}
  fs.readFile(file,(err,body)=>{res.writeHead(err?404:200,{'Content-Type':mime[path.extname(file)]||'application/octet-stream'});res.end(err?'Missing':body);});
});
(async()=>{
 await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
 const base='http://127.0.0.1:'+server.address().port;let browser;
 try{
  browser=await chromium.launch({headless:true,executablePath:process.argv[3]||undefined});
  const context=await browser.newContext({viewport:{width:1440,height:1100}}),page=await context.newPage(),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await context.route('**/*',route=>{const url=route.request().url();if(!url.startsWith(base+'/')&&!/^(data:|blob:)/.test(url)){errors.push('External: '+url);return route.abort();}return route.continue();});
  await page.goto(base+'/index.html');await page.locator('.lesson-card').first().waitFor();
  const go=async hash=>{await page.evaluate(h=>{location.hash=h;},hash);await page.waitForTimeout(35);};
  for(const width of [1440,390]){
   await page.setViewportSize({width,height:1100});await page.waitForTimeout(300);
   for(const d of data.diagrams){
    await go('#diagrams?model='+d.id);
    assert.equal(await page.locator('.diagram-node').count(),d.nodes.length,d.id);
    assert.equal(await page.locator('.diagram-edge').count(),d.edges.length,d.id);
    assert.equal(await page.locator('#diagram-model').inputValue(),d.id);
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true,'Overflow: '+d.id+' @'+width);
    const overflow=await page.locator('.diagram-node').evaluateAll(nodes=>nodes.filter(n=>{
      const rect=n.querySelector('rect').getBBox();
      return [...n.querySelectorAll('text')].some(t=>{const b=t.getBBox();return b.x<0||b.y<0||b.x+b.width>rect.width||b.y+b.height>rect.height;});
    }).map(n=>n.dataset.node));
    assert.deepEqual(overflow,[],'Text outside block: '+d.id);
    const chosen=d.nodes[d.nodes.length-1];
    const node=page.locator('[data-node="'+chosen.id+'"]');
    await node.scrollIntoViewIfNeeded();await node.click();
    assert.equal(await node.getAttribute('aria-pressed'),'true');
    assert.equal(await page.locator('.diagram-detail p').textContent(),chosen.detail);
   }
  }
  await go('#theory?lesson=3&topic=resnet');
  assert.equal(await page.locator('[data-diagram-picker] option').count(),4);
  await page.locator('[data-diagram-picker]').selectOption('densenet');
  assert.equal(await page.locator('.architecture-figure').getAttribute('data-model'),'densenet');
  const target=page.locator('[data-node="cat"]');await target.focus();await target.press('Enter');
  assert.equal(await target.getAttribute('aria-pressed'),'true');
  await page.setViewportSize({width:1440,height:1100});await page.waitForTimeout(300);
  fs.mkdirSync(out,{recursive:true});
  for(const theme of ['neon-dark','neon-light','classic']){
   await page.locator('#theme-select').selectOption(theme);
   await go('#theory?lesson=4&topic=bert');
   if(await page.locator('[data-diagram-expand]').getAttribute('aria-expanded')!=='true')await page.locator('[data-diagram-expand]').click();
   assert.equal(await page.locator('[data-diagram-expand]').getAttribute('aria-expanded'),'true');
   const legend=await page.locator('.diagram-legend i').evaluateAll(items=>items.map(i=>getComputedStyle(i).position));
   assert.deepEqual(legend,['static','static','static','static']);
   await page.locator('.architecture-schemes').screenshot({path:path.join(out,'bert-scheme-'+theme+'.png')});
  }
  await go('#diagrams?lesson=4&model=coca');
  await page.screenshot({path:path.join(out,'diagram-catalog.png'),fullPage:true});
  await page.locator('#diagram-lesson').selectOption('5');await page.waitForTimeout(60);
  assert.ok(await page.locator('#diagram-model option').count()>5);
  await go('#practice?architecture=bert&type=number&count=5');await page.locator('#practice-form button[type=submit]').click();await page.locator('#answer-form').waitFor();
  await page.locator('#answer-text').fill('300');const saved=await page.evaluate(()=>JSON.parse(JSON.stringify(state)));
  await go('#diagrams?model=bert');await page.locator('[data-node="emb"]').click();await page.reload();await page.locator('.diagram-node').first().waitFor();
  assert.deepEqual(await page.evaluate(()=>JSON.parse(JSON.stringify(state))),saved);
  assert.deepEqual(errors,[]);
  console.log('Diagram UI passed: 40 schemes at desktop/mobile, text bounds, node selection, keyboard, variants, filters, 3 themes, progress preservation; no external requests.');
 }finally{if(browser)await browser.close();await new Promise(resolve=>server.close(resolve));}
})().catch(e=>{console.error(e);process.exitCode=1;});
