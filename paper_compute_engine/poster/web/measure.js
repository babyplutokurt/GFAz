// Report how tall the poster content actually is vs. the 45in page,
// and flag any element whose box escapes the page horizontally.
const puppeteer = require('puppeteer');
const path = require('path');

(async () => {
  const src = process.argv[2] || 'poster.html';
  const browser = await puppeteer.launch({args: ['--no-sandbox']});
  const page = await browser.newPage();
  await page.setViewport({width: 4032, height: 4320, deviceScaleFactor: 1});
  await page.goto('file://' + path.resolve(src), {waitUntil: 'networkidle0'});
  await page.evaluateHandle('document.fonts.ready');

  const report = await page.evaluate(() => {
    const DPI = 96;
    const body = document.body;
    const pageH = 45 * DPI, pageW = 42 * DPI;
    const out = {
      pageH, contentH: body.scrollHeight,
      overflowIn: +((body.scrollHeight - pageH) / DPI).toFixed(2),
      sections: [], clipped: []
    };
    for (const sel of ['header', '.stats', 'main', 'footer']) {
      const el = document.querySelector(sel);
      if (el) out.sections.push({sel, h: +(el.getBoundingClientRect().height / DPI).toFixed(2)});
    }
    document.querySelectorAll('.col > *').forEach(el => {
      const r = el.getBoundingClientRect();
      const label = (el.querySelector('.lbl,h2,.big')?.textContent || el.className || el.tagName)
        .trim().slice(0, 44);
      out.sections.push({sel: '  ' + label, h: +(r.height / DPI).toFixed(2)});
    });
    // horizontal escapes
    document.querySelectorAll('svg.chart, table, .note, .callout').forEach(el => {
      const r = el.getBoundingClientRect();
      if (r.right > pageW - 40 || r.left < 20) {
        out.clipped.push({el: el.className || el.tagName,
                          right: Math.round(r.right), pageW});
      }
    });
    return out;
  });

  console.log(`page height   : ${(report.pageH/96).toFixed(2)} in`);
  console.log(`content height: ${(report.contentH/96).toFixed(2)} in`);
  console.log(`OVERFLOW      : ${report.overflowIn} in` +
              (report.overflowIn > 0 ? '   <-- must reach <= 0' : '   OK'));
  console.log('\nsection heights (in):');
  report.sections.forEach(s => console.log(`  ${String(s.h).padStart(6)}  ${s.sel}`));
  if (report.clipped.length) {
    console.log('\nhorizontally clipped:');
    report.clipped.forEach(c => console.log('  ', JSON.stringify(c)));
  }
  await browser.close();
})();
