// Offline rendering and isolated browser tests; no access to the user's study storage.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),http=require('node:http');
const root=path.resolve(__dirname,'../dist'),out=path.resolve(__dirname,'../qa');
const sandbox={window:{}};vm.runInNewContext(fs.readFileSync(path.join(root,'content.js'),'utf8'),sandbox);
const data=sandbox.window.CV_CONTENT;
const katex=require('../dist/vendor/katex.min.js');
let formulaCount=0;
for(const t of data.topics){
  if(t.formula){katex.renderToString(t.formula,{throwOnError:true,strict:false});formulaCount++;}
  for(const match of t.mathNotes.matchAll(/\$\$([\s\S]*?)\$\$|\$([^$\n]+?)\$/g)){
    katex.renderToString(match[1]||match[2],{throwOnError:true,strict:false});formulaCount++;
  }
}
for(const q of data.questions)if(q.calculation){katex.renderToString(q.calculation,{throwOnError:true,strict:false});formulaCount++;}
function checkProse(text){for(const m of text.matchAll(/\$\$([\s\S]*?)\$\$|\$([^$\n]+?)\$/g)){katex.renderToString(m[1]||m[2],{throwOnError:true,strict:false});formulaCount++;}}
for(const t of data.topics){checkProse(t.walkthrough.given);for(const s of t.walkthrough.steps)checkProse(s.text);}
for(const d of data.diagrams)for(const n of d.nodes)for(const s of n.operations||[])checkProse(s.text);
console.log('KaTeX syntax passed:',formulaCount,'formulas across 65 topics and 93 worked answers');
if(process.argv.includes('--formulas-only'))process.exit(0);
const {chromium}=require(process.argv[2]||'playwright');
const mime={'.html':'text/html','.js':'text/javascript','.css':'text/css','.woff2':'font/woff2','.woff':'font/woff','.png':'image/png'};
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
    const context=await browser.newContext({viewport:{width:1440,height:1100}});
    // Test copy controls without touching the OS clipboard.
    await context.addInitScript(()=>{Object.defineProperty(navigator,'clipboard',{value:{writeText:async text=>{window.testCopied=text;}}});});
    const page=await context.newPage(),errors=[];
    page.on('pageerror',e=>errors.push(e.message));
    await context.route('**/*',route=>{
      const url=route.request().url();
      if(!url.startsWith(base+'/')&&!/^(data:|blob:)/.test(url)){errors.push('External request: '+url);return route.abort();}
      return route.continue();
    });
    await page.goto(base+'/index.html');await page.locator('.lesson-card').first().waitFor();
    await page.evaluate(()=>document.fonts.ready);
    const go=async hash=>{await page.evaluate(h=>{location.hash=h;},hash);await page.waitForTimeout(35);};
    for(const width of [1440,390]){
      await page.setViewportSize({width,height:1100});await page.waitForTimeout(300);
      for(const t of data.topics){
        await go('#theory?topic='+t.id+'&lesson='+t.lesson);
        assert.equal(await page.locator('.math-notes').count(),1,t.id);
        assert.equal(await page.locator('.implementation code').textContent(),t.implementation.code,t.id);
        assert.equal(await page.locator('.katex-error').count(),0,t.id);
        assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true,'Overflow: '+t.id+' @ '+width);
      }
    }
    await page.setViewportSize({width:1440,height:1100});await page.waitForTimeout(300);
    await go('#theory?topic=attention&lesson=4');
    await page.locator('[data-copy-code]').click();
    assert.equal(await page.evaluate(()=>window.testCopied),data.topics.find(t=>t.id==='attention').implementation.code);
    await page.evaluate(()=>{navigator.clipboard.writeText=async()=>{throw Error('Unavailable');};});
    await page.locator('[data-copy-code]').click();
    assert.equal(await page.evaluate(()=>String(window.getSelection())),data.topics.find(t=>t.id==='attention').implementation.code);
    await page.evaluate(()=>window.getSelection().removeAllRanges());
    fs.mkdirSync(out,{recursive:true});
    for(const theme of ['neon-dark','neon-light','classic']){
      await page.locator('#theme-select').selectOption(theme);
      assert.equal(await page.locator('.code-example code').evaluate(e=>getComputedStyle(e).backgroundColor),'rgba(0, 0, 0, 0)');
      await page.locator('.math-notes').screenshot({path:path.join(out,'math-'+theme+'.png')});
      await page.locator('.code-example').screenshot({path:path.join(out,'code-'+theme+'.png')});
    }
    await go('#practice?architecture=swin&type=number&count=5');
    await page.locator('#practice-form button[type=submit]').click();await page.locator('#answer-form').waitFor();
    const q=await page.evaluate(()=>getQ());
    await page.locator('#answer-text').fill(String(q.value));await page.locator('#answer-form button[type=submit]').click();
    assert.ok(await page.locator('.feedback>.formula .katex').count());
    assert.equal(await page.locator('.math-review .implementation').isVisible(),false);
    await page.locator('.math-review>summary').click();
    assert.equal(await page.locator('.math-review .implementation').isVisible(),true);
    const saved=await page.evaluate(()=>JSON.parse(JSON.stringify(state)));
    await page.reload();await page.locator('.feedback').waitFor();
    assert.deepEqual(await page.evaluate(()=>JSON.parse(JSON.stringify(state))),saved);
    await go('#search?q=window_partition');assert.ok(await page.locator('.search-hit').count());
    assert.deepEqual(errors,[]);
    console.log('Math browser checks passed: all 65 topics at desktop/mobile, code copy/fallback, 3 themes, worked answers, review details, persistence, code search.');
  }finally{if(browser)await browser.close();await new Promise(resolve=>server.close(resolve));}
})().catch(e=>{console.error(e);process.exitCode=1;});
