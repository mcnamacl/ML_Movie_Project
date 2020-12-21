import requests

TMDB_API_key = "e1b7bcccdd88c2cbdd4d3dfb39a4733f"
TMDB_req = "https://api.themoviedb.org/3/find/movie/"
allResults = []
ids = []
titles = []

def return_results(site, params):
    r = requests.get(url=site, params=params)
    return r.json()

def add_json_ids(site, params):
    r = return_results(site, params)
    # totalPages = r["total_pages"]
    # should use totalPages but using 5 instead for now as 500 takes a while
    year_ids = []
    year_titles = []
    for page in range(1, 100 + 1):
        params["page"] = page
        r = return_results(site, params)
        for result in r["results"]:
            year_ids.append(result["id"])
            year_titles.append(result["original_title"])
    titles.append(year_titles)
    ids.append(year_ids)

def add_json_results(site, params, features, title):
    r = return_results(site, params)
    if is_useful(r):
        omdb = requests.get("http://www.omdbapi.com/?apikey=1df54ea&t=" + title)
        if omdb.status_code == 200:
            omdb = omdb.json()
            i = 0
            for feature in features:
                try:
                    allResults[i]
                except IndexError:
                    allResults.append([])

                if feature == "imdbRating" or feature == "imdbVotes":
                    if feature in omdb:
                        new_sample = omdb[feature]
                        if(new_sample == "N/A"):
                                new_sample = 0.0 
                    else:
                        new_sample = 0.0 
                else:
                    new_sample = r[feature]
                    if feature == "production_companies" or feature == "genres":
                        new_sample = return_names(r[feature])
                    if feature == "release_date":
                        new_sample = return_month(r[feature])
                allResults[i].append(new_sample)
                i = i + 1

def is_useful(r):
    if not "budget" in r or not "revenue" in r:
        return False
    return r["budget"] != 0 and r["revenue"] != 0

def return_names(info):
    companies = []
    for i in range(0, len(info)):
        companies.append(info[i]["name"])
    return companies

def return_month(date):
    return date.split("-")[1]

def write_to_file(results):
    with open("CSV_files/test_pre_processing.csv", "a+") as f:
        for x in zip(*results):
            f.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\n".format(*x))

if __name__ == "__main__":
    # Getting movie IDs.
    site = "http://api.themoviedb.org/3/discover/movie"
    #years = [2015, 2016, 2017]
    years = [2018]
    features = ["id"]
    for year in years:
        params = {
            "api_key": "e1b7bcccdd88c2cbdd4d3dfb39a4733f",
            "primary_release_year": year,
            "page": 1,
        }
        add_json_ids(site, params)

    # Getting information for each movie.
    features = [
        "revenue",
        "budget",
        "release_date",
        "production_companies",
        "genres",
        "original_language",
        "runtime",
        "imdbRating",
        "imdbVotes"
    ]
    for i in range(0, len(years)):
        allResults = []
        index = 0
        for id in ids[i]:
            site = "http://api.themoviedb.org/3/movie/" + str(id)
            params = {"api_key": "e1b7bcccdd88c2cbdd4d3dfb39a4733f"}
            add_json_results(site, params, features, titles[i][index])
            index+=1
        write_to_file(allResults)
