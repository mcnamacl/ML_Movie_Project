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
    for page in range(1, 100 + 1):
        params["page"] = page
        r = return_results(site, params)
        for result in r["results"]:
            year_ids.append(result["id"])
            titles.append(result["original_title"])
    ids.append(year_ids)

def add_json_results(site, params, features):
    r = return_results(site, params)
    if is_useful(r):
        i = 0
        for feature in features:
            try:
                allResults[i]
            except IndexError:
                allResults.append([])

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
    with open("original_data.csv", "a+") as f:
        for x in zip(*results):
            f.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\n".format(*x))

if __name__ == "__main__":
    # Getting movie IDs.
    site = "http://api.themoviedb.org/3/discover/movie"
    years = [2015, 2016, 2017]
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
    ]
    for i in range(0, len(years)):
        allResults = []
        for id in ids[i]:
            site = "http://api.themoviedb.org/3/movie/" + str(id)
            params = {"api_key": "e1b7bcccdd88c2cbdd4d3dfb39a4733f"}
            add_json_results(site, params, features)
        write_to_file(allResults)
