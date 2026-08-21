const puppeteer = require('puppeteer');
(async () => {
  const [src, out] = process.argv.slice(2);
  const browser = await puppeteer.launch({args:['--no-sandbox','--font-render-hinting=none']});
  const page = await browser.newPage();
  await page.goto('file://' + require('path').resolve(src), {waitUntil:'networkidle0'});
  await page.evaluateHandle('document.fonts.ready');
  await page.pdf({path: out, preferCSSPageSize: true, printBackground: true});
  await browser.close();
})();
