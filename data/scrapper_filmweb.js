const puppeteer = require("puppeteer");
const ObjectsToCsv = require("objects-to-csv");

(async () => {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();

  // crawl all links to pages with movies details
  const allMoviePageLinks = [];
  for (let i = 1; i <= 1000; i++) {
    console.log(i);
    await page.goto(`https://www.filmweb.pl/films/search?page=${i}`);
    const linksOnPage = await page.evaluate(() => {
      const aLinks = document.querySelectorAll("a.poster__link");
      return Array.from(aLinks).map((a) => a.href);
    });
    allMoviePageLinks.push(...linksOnPage);
  }

  console.log(allMoviePageLinks.length);
  // console.log(allMoviePageLinks)

  const moviesData = [];
  const errors = [];

  // fetch movies details from all crawled links
  for (const link of allMoviePageLinks) {
    console.log(link);
    await page.goto(link);

    try {
      const movieDetail = await page.evaluate(() => {
        const title = document.querySelector(
          ".filmCoverSection__title"
        ).innerHTML;
        const originalTitle = document.querySelector(
          ".filmCoverSection__originalTitle"
        )?.innerHTML;
        const description = document.querySelector(
          "[itemprop=description]"
        ).innerHTML;
        const year = document.querySelector(
          ".filmCoverSection__year"
        ).innerHTML;
        const genre = document.querySelector("[itemprop=genre]").innerText;
        const movie = {
          title: title,
          originalTitle: originalTitle,
          year: year,
          description: description,
          genre: genre,
        };
        return movie;
      });
      moviesData.push(movieDetail);
    } catch (err) {
      errors.push(link);
      console.log("Can't fetch", link);
    }
  }
  await browser.close();

  // log urls that were not fetched
  console.log("Errors", errors);
  // console.log(moviesData)

  // save data to csv
  const csv = new ObjectsToCsv(moviesData);
  await csv.toDisk("./filmweb.csv");
})();
