#!/usr/bin/env python3
"""
Generate glove-vectors.js from GloVe 6B 50-dimensional vectors.

Outputs the top 10 000 words by frequency PLUS any word from the curated
Places / Food / People seed lists found anywhere in the full 400 k vocabulary.

Steps:
  1. Place glove.6B.50d.txt in this directory (or let the script download it).
  2. Run:  python3 generate-glove.py
  Output: glove-vectors.js  (~4-5 MB, L2-normalised)
"""

import json, math, os, sys, urllib.request, zipfile, io

GLOVE_TXT  = "glove.6B.50d.txt"
OUTPUT_JS  = "glove-vectors.js"
MAX_WORDS  = 10_000   # top-frequency words always included
DIMS       = 50
GLOVE_URL  = "https://nlp.stanford.edu/data/glove.6B.zip"

# ---------------------------------------------------------------------------
# Curated seed words — included regardless of frequency rank
# ---------------------------------------------------------------------------

SEED_PLACES = {
    # Countries
    'afghanistan','albania','algeria','angola','argentina','armenia','australia',
    'austria','azerbaijan','bahrain','bangladesh','belarus','belgium','bolivia',
    'bosnia','botswana','brazil','brunei','bulgaria','cambodia','cameroon',
    'canada','chad','chile','china','colombia','congo','croatia','cuba','cyprus',
    'czech','denmark','ecuador','egypt','england','eritrea','ethiopia','fiji',
    'finland','france','gabon','gambia','georgia','germany','ghana','greece',
    'guatemala','guinea','guyana','haiti','honduras','hungary','iceland','india',
    'indonesia','iran','iraq','ireland','israel','italy','jamaica','japan',
    'jordan','kazakhstan','kenya','kosovo','kuwait','kyrgyzstan','laos','latvia',
    'lebanon','lesotho','liberia','libya','liechtenstein','lithuania','luxembourg',
    'malawi','malaysia','maldives','mali','malta','mauritania','mauritius',
    'mexico','moldova','monaco','mongolia','montenegro','morocco','mozambique',
    'myanmar','namibia','nepal','netherlands','nicaragua','niger','nigeria',
    'norway','oman','pakistan','palau','palestine','panama','paraguay','peru',
    'philippines','poland','portugal','qatar','romania','russia','rwanda',
    'samoa','saudi','scotland','senegal','serbia','singapore','slovakia',
    'slovenia','somalia','spain','sudan','suriname','swaziland','sweden',
    'switzerland','syria','taiwan','tajikistan','tanzania','thailand','togo',
    'trinidad','tunisia','turkey','turkmenistan','uganda','ukraine','uruguay',
    'uzbekistan','venezuela','vietnam','wales','yemen','zambia','zimbabwe',
    'korea','macedonia',
    # US States
    'alabama','alaska','arizona','arkansas','california','colorado','connecticut',
    'delaware','florida','hawaii','idaho','illinois','indiana','iowa','kansas',
    'kentucky','louisiana','maine','maryland','massachusetts','michigan',
    'minnesota','mississippi','missouri','montana','nebraska','nevada','ohio',
    'oklahoma','oregon','pennsylvania','tennessee','texas','utah','vermont',
    'virginia','wisconsin','wyoming','hampshire','jersey','carolina','dakota',
    # Canadian provinces
    'ontario','quebec','alberta','brunswick','saskatchewan','newfoundland',
    'manitoba','columbia','nova',
    # Australian states
    'queensland','tasmania',
    # Geographic regions & continents
    'caribbean','mediterranean','scandinavia','caucasus','appalachia',
    'amazon','siberia','patagonia','sahel','balkans','anatolia',
    'levant','mesopotamia','maghreb','serengeti',
    # Mountain ranges & peaks
    'himalayas','andes','alps','pyrenees','rockies','urals',
    'everest','kilimanjaro','fuji','olympus','blanc','matterhorn','denali',
    # Deserts
    'sahara','gobi','mojave','atacama','kalahari','namib','arabian','negev',
    # Oceans, seas & major waterways
    'atlantic','pacific','arctic','antarctic','mediterranean','caribbean',
    'caspian','bering',
    # Rivers
    'nile','amazon','yangtze','ganges','mekong','danube','rhine','volga',
    'tigris','euphrates','thames','seine','tiber','indus','orinoco',
    'zambezi','congo','niger','mekong','brahmaputra',
    # Lakes
    'superior','michigan','huron','erie','titicaca','baikal','tanganyika',
    # Islands
    'sicily','sardinia','corsica','java','sumatra','borneo','tasmania',
    'madagascar','greenland','crete','bahamas','barbados','luzon','mindanao',
    'iceland','ireland','britain','zealand','papua',
    # World capitals & major cities
    'amsterdam','amman','ankara','athens','atlanta','auckland','baghdad',
    'baku','bangkok','barcelona','beijing','beirut','belgrade','berlin',
    'bogota','boston','brasilia','bratislava','brussels','bucharest','budapest',
    'buenos','cairo','caracas','casablanca','chicago','colombo','copenhagen',
    'dallas','damascus','delhi','denver','detroit','dhaka','doha','dubai',
    'dublin','edinburgh','frankfurt','geneva','hanoi','harare','havana',
    'helsinki','houston','islamabad','istanbul','jakarta','johannesburg',
    'kabul','kampala','karachi','kathmandu','khartoum','kiev','kinshasa',
    'kyoto','lahore','lima','lisbon','london','luanda','lusaka','madrid',
    'manila','maputo','melbourne','miami','milan','minsk','mogadishu',
    'montreal','moscow','mumbai','muscat','nairobi','nicosia','oslo',
    'ottawa','paris','prague','pretoria','pyongyang','quito','rabat',
    'reykjavik','riga','rio','riyadh','rome','santiago','sarajevo',
    'seattle','seoul','shanghai','singapore','sofia','stockholm','sydney',
    'taipei','tashkent','tbilisi','tehran','tokyo','toronto','tripoli',
    'tunis','ulaanbaatar','valletta','vancouver','vienna','vilnius',
    'warsaw','washington','wellington','zagreb','zurich',
    # European cities
    'milan','lyon','bilbao','cannes','genoa','porto','cork','edinburgh',
    'valencia','seville','naples','florence','venice','palermo','barcelona',
    'antwerp','bruges','ghent','rotterdam','hamburg','munich','cologne',
    'frankfurt','dresden','nuremberg','leipzig','gdansk','krakow','wroclaw',
    'poznan','lodz','lviv','odessa','kharkiv','donetsk','tirana','yerevan',
    'bishkek','lausanne','andorra',
    # Japanese cities
    'osaka','nagoya','kobe','fukuoka','yokohama','sendai','hiroshima',
    'nagasaki','sapporo',
    # Chinese cities
    'guangzhou','shenzhen','wuhan','nanjing','tianjin','chongqing',
    'hangzhou','macau','qingdao','dalian','harbin','changsha','kunming',
    'xian','jinan','zhengzhou','shenyang','lanzhou','nanchang',
    # South Asian cities
    'bangalore','hyderabad','pune','ahmedabad','chittagong','lahore',
    'karachi','islamabad','kathmandu','colombo','dhaka','amritsar','jaipur',
    # Southeast Asian cities
    'phnom','mandalay','yangon','cebu','davao','quezon','manila','jakarta',
    'kuala','penang','medan','surabaya',
    # African cities
    'abuja','accra','addis','abidjan','dakar','freetown','kano','kumasi',
    'kigali','kinshasa','mombasa','dar','kampala','harare','bulawayo',
    'durban','nairobi','lagos','johannesburg','cape','pretoria',
    'casablanca','tunis','tripoli','khartoum','mogadishu','juba',
    'lome','bamako','conakry','ouagadougou','niamey','bangui','libreville',
    'brazzaville','luanda','lusaka','windhoek','gaborone','maputo',
    'lilongwe','antananarivo',
    # Americas cities
    'guadalajara','monterrey','medellín','recife','fortaleza','salvador',
    'curitiba','manaus','asuncion','montevideo','guayaquil','quito',
    'lima','bogota','cali','caracas','havana','managua','tegucigalpa',
    'panama','sanjose',
    # Middle East & Central Asia cities
    'mecca','medina','jeddah','riyadh','doha','muscat','amman','beirut',
    'aleppo','haifa','tel','almaty','tashkent','bishkek','ashgabat',
    'dushanbe','kabul','baku',
    # Other notable cities
    'brisbane','perth','auckland','christchurch','wellington',
    'cairo','rabat','tunis','algiers',
    # More secondary cities
    'albuquerque','anchorage','omaha','raleigh','spokane','tulsa',
    'laredo','lubbock','plano','mesa','scottsdale','chandler','tempe',
    'minneapolis','portland','denver','atlanta','miami','boston',
    'detroit','pittsburgh','cleveland','cincinnati','milwaukee',
    'memphis','nashville','louisville','richmond','baltimore',
    'montreal','vancouver','calgary','edmonton','ottawa','winnipeg',
    'abuja','accra','addis','kano','kigali','kinshasa',
    'sarajevo','skopje','pristina','chisinau','minsk','kyiv',
    'dortmund','essen','hannover','stuttgart','bremen',
    'marseille','toulouse','nantes','strasbourg','montpellier',
    'zaragoza','malaga','bilbao','alicante',
    'naples','palermo','bologna','florence','venice','genoa',
    'porto','braga','funchal',
    'rotterdam','eindhoven','utrecht','bruges','ghent','antwerp',
    'oslo','bergen','stavanger','trondheim',
    'stockholm','gothenburg','malmo',
    'helsinki','tampere','turku',
    'copenhagen','aarhus','aalborg',
    'warsaw','lodz','krakow','wroclaw','poznan','gdansk',
    'budapest','debrecen','miskolc',
    'prague','brno','ostrava',
    'bratislava','kosice',
    'zagreb','split','rijeka',
    'belgrade','novi','nis',
    'bucharest','cluj','timisoara','iasi','constanta',
    'sofia','plovdiv','varna',
    'athens','thessaloniki','patras',
    'istanbul','ankara','izmir','bursa','adana','konya',
    'kiev','kharkiv','odessa','lviv','donetsk','dnipro',
    'minsk','vitebsk','grodno',
    'riga','daugavpils',
    'tallinn','tartu',
    'vilnius','kaunas',
    'moscow','saint','petersburg','novosibirsk','yekaterinburg',
    'nizhny','kazan','omsk','chelyabinsk','samara','ufa','rostov',
    'volgograd','vladivostok','sochi',
    'tbilisi','batumi','kutaisi',
    'yerevan','gyumri',
    'baku','ganja','sumgait',
    'astana','almaty','shymkent',
    'tashkent','samarkand','namangan',
    'bishkek','osh',
    'dushanbe','khujand',
    'ashgabat','turkmenabat',
    'kabul','kandahar','herat','mazar',
}

SEED_FOOD = {
    # Staples
    'rice','bread','pasta','noodles','wheat','corn','potato','tomato','onion',
    'garlic','pepper','salt','sugar','oil','olive','butter','cream','milk',
    'eggs','cheese','flour','yeast','maize','cassava','taro','yam','plantain',
    # Proteins — meat
    'chicken','beef','pork','lamb','turkey','duck','goose','venison','quail',
    'bacon','ham','steak','sausage','salami','chorizo','pepperoni',
    'prosciutto','pancetta','mortadella','lard','jerky',
    # Proteins — seafood
    'fish','salmon','tuna','cod','shrimp','lobster','crab','oyster',
    'sardine','trout','anchovy','mackerel','herring','squid','octopus',
    'clam','mussel','scallop','prawn','eel','caviar',
    # Fruits — common
    'apple','banana','orange','lemon','grape','strawberry','blueberry',
    'raspberry','blackberry','cherry','peach','plum','pear','mango',
    'pineapple','watermelon','coconut','kiwi','avocado','papaya','guava',
    'lychee','jackfruit','durian','persimmon','clementine','tangerine',
    'grapefruit','pomelo','kumquat','fig','date','apricot','nectarine',
    # Fruits — less common but notable
    'cranberry','pomegranate','mulberry','quince','elderberry','loquat',
    'rambutan','longan','starfruit','dragonfruit','passionfruit',
    'gooseberry','currant','boysenberry','huckleberry','tamarind',
    'mandarin','citron','bergamot','yuzu','physalis','feijoa',
    # Vegetables
    'broccoli','spinach','carrot','celery','cucumber','zucchini','eggplant',
    'asparagus','cauliflower','lettuce','cabbage','radish','beetroot',
    'turnip','parsnip','pumpkin','squash','mushroom','fennel','leek',
    'artichoke','arugula','kale','sweetcorn','okra','endive','chicory',
    'kohlrabi','daikon','jicama','shallot','chive','sorrel','samphire',
    'watercress','bok','morel','truffle','chanterelle','portobello',
    # Nuts, seeds & legumes
    'almond','walnut','cashew','peanut','pistachio','pecan','hazelnut',
    'chestnut','macadamia','brazil','pine',
    'quinoa','sesame','flaxseed','chia','sunflower','pumpkin',
    'lentil','chickpea','soybean','edamame','bean','pea',
    # Fast food & street food
    'burger','hotdog','fries','sandwich','kebab',
    # Cheeses
    'parmesan','mozzarella','cheddar','brie','feta','ricotta','gouda',
    'gruyere','manchego','provolone','camembert','roquefort','stilton',
    'pecorino','emmental','edam','havarti','asiago','fontina','colby',
    'gorgonzola','taleggio','halloumi','paneer',
    # Baked & pastry
    'pizza','cake','cookie','brownie','muffin','donut','pie','tart',
    'waffle','pancake','croissant','baguette','pretzel','bagel','scone',
    'biscuit','cracker','brioche','focaccia','calzone','ciabatta',
    'sourdough','cornbread','pumpernickel','lavash','pita','flatbread',
    'challah','naan','roti','chapati','paratha','tortilla',
    # Desserts
    'tiramisu','gelato','sorbet','mousse','pudding','custard','caramel',
    'fudge','toffee','chocolate','meringue','pavlova','flan','crepe',
    'baklava','cannoli','macaron','profiterole','eclair','strudel',
    'cheesecake','trifle','cobbler','crumble','souffle',
    # Asian dishes
    'sushi','ramen','tempura','sashimi','miso','kimchi','bibimbap',
    'pho','banh','satay','rendang','pad','curry','biryani','samosa',
    'pakora','dal','tikka','masala','korma','vindaloo','saag','dosa','idli',
    'kebab','shawarma','falafel','hummus','gyros','moussaka',
    'dumpling','gyoza','bulgogi','yakitori','teriyaki','edamame','tofu',
    'sukiyaki','udon','soba','takoyaki','okonomiyaki','tonkatsu',
    'katsu','onigiri','mochi','matcha','wasabi',
    'congee','dimsum','wonton','chow','hotpot','mapo','xiaolongbao',
    'laksa','nasi','mee','char','rojak','bak',
    'tteokbokki','japchae','sundubu','samgyeopsal','galbi','jjigae',
    # Middle Eastern & Mediterranean
    'paella','gazpacho','tapas','churros','empanada',
    'risotto','lasagna','spaghetti','ravioli','fettuccine','linguine',
    'penne','gnocchi','macaroni','pesto','bruschetta','carpaccio',
    'antipasto','prosciutto','bruschetta','focaccia','calzone',
    'couscous','tagine','harissa','tahini','tabbouleh','dolma',
    'baba','kibbeh','mansaf','shakshuka',
    # Mexican & Latin
    'taco','burrito','quesadilla','nacho','enchilada','tamale','pozole',
    'guacamole','salsa','mole','ceviche','arepa','empanada','churrasco',
    'chimichanga','fajita','carnitas','elote',
    # Eastern European
    'pierogi','borscht','stroganoff','goulash','schnitzel','bratwurst',
    'fondue','knish','latke','blini','pelmeni','varenyky','kvass',
    # Condiments & spices
    'ketchup','mustard','mayo','vinegar','soy','sauce','gravy','dressing',
    'syrup','honey','jam','marmalade','cinnamon','vanilla','saffron',
    'paprika','cumin','turmeric','oregano','basil','thyme','rosemary',
    'mint','parsley','cilantro','ginger','chili','clove','nutmeg',
    'coriander','jalapeno','cardamom','anise','fennel','tarragon',
    'horseradish','wasabi','zaatar','sumac','baharat','harissa',
    # Drinks — hot
    'coffee','espresso','cappuccino','latte','americano','mocha','macchiato',
    'tea','matcha','chai','cocoa',
    # Drinks — beer & cider
    'beer','lager','ale','stout','porter','guinness','pilsner','cider',
    'mead','sake','soju',
    # Drinks — wine & spirits
    'wine','champagne','prosecco','cava','sangria','mulled',
    'vodka','whiskey','rum','gin','bourbon','tequila','brandy',
    'cognac','armagnac','calvados','grappa','schnapps','vermouth',
    'absinthe','mezcal','ouzo','raki','arak','pisco','cachaca',
    'chianti','sauvignon','riesling','merlot','chardonnay','shiraz',
    'pinot','cabernet','beaujolais','burgundy','bordeaux','rioja',
    'sherry','port','madeira','marsala','amaretto','baileys',
    # Drinks — soft
    'juice','soda','smoothie','milkshake','lemonade','horchata',
    'lassi','kombucha','kefir','ayran','tamarind',
    # Dairy
    'yogurt','butter','cream','milk','ghee','kefir',
    # Other
    'acai','poke','granola','muesli','porridge','oatmeal',
}

SEED_PEOPLE = {
    # Scientists & mathematicians
    'einstein','newton','darwin','galileo','curie','hawking','tesla','faraday',
    'bohr','planck','heisenberg','feynman','turing','pasteur','mendel',
    'crick','oppenheimer','fermi','rutherford','hubble','sagan',
    'lavoisier','celsius','fahrenheit','kelvin','volta','ampere','hertz',
    'pascal','leibniz','euler','gauss','fibonacci','archimedes','copernicus',
    # Philosophers
    'aristotle','plato','socrates','kant','nietzsche','descartes','locke',
    'hume','voltaire','rousseau','marx','freud','jung','confucius',
    'lao','mencken','emerson','thoreau','sartre','camus','de',
    # Writers & poets
    'shakespeare','dante','goethe','dostoevsky','tolstoy','kafka',
    'hemingway','steinbeck','orwell','dickens','twain','austen','woolf',
    'joyce','homer','virgil','cicero','cervantes','hugo','balzac',
    'flaubert','zola','proust','borges','marquez','neruda','paz',
    'keats','shelley','byron','blake','wordsworth','tennyson','browning',
    'chaucer','milton','poe','whitman',
    # US Presidents
    'washington','adams','jefferson','madison','monroe','jackson','harrison',
    'polk','lincoln','grant','hayes','garfield','cleveland','mckinley',
    'roosevelt','taft','wilson','harding','coolidge','hoover','truman',
    'eisenhower','kennedy','johnson','nixon','ford','carter','reagan',
    'bush','clinton','obama','trump','biden','fillmore',
    # World leaders & historical figures
    'napoleon','caesar','cleopatra','augustus','charlemagne','genghis',
    'saladin','columbus','magellan','cortez','drake','wellington','bismarck',
    'garibaldi','bolivar','mandela','gandhi','nehru','mao','stalin','hitler',
    'mussolini','churchill','thatcher','franco','pinochet','castro',
    'ataturk','khomeini','nkrumah','kenyatta','nyerere','lumumba',
    # Arts & music
    'mozart','beethoven','bach','handel','chopin','brahms','liszt','wagner',
    'verdi','puccini','tchaikovsky','debussy','stravinsky','mahler',
    'picasso','michelangelo','rembrandt','raphael','leonardo','caravaggio',
    'monet','manet','renoir','cezanne','dali','warhol','pollock',
    'rodin','matisse','klimt','munch','kandinsky','chagall',
    # Tech & business
    'gates','jobs','zuckerberg','bezos','musk','buffett','rockefeller',
    'ford','carnegie','morgan','edison','bell',
    # Religion
    'buddha','jesus','muhammad','moses','abraham','augustine','luther','calvin',
    # Modern & royalty
    'victoria','elizabeth','henry','alexander','hamilton','franklin',
    'mandela','oprah','teresa','diana','malala','dalai',
}

ALL_SEEDS = SEED_PLACES | SEED_FOOD | SEED_PEOPLE


def download_glove():
    print(f"Downloading GloVe from {GLOVE_URL} (~170 MB) …")
    buf = io.BytesIO()
    with urllib.request.urlopen(GLOVE_URL) as r:
        total = int(r.headers.get("Content-Length", 0))
        done  = 0
        while chunk := r.read(1 << 16):
            buf.write(chunk)
            done += len(chunk)
            if total:
                pct = done * 100 // total
                print(f"\r  {pct}% ({done // 1024 // 1024} MB)", end="", flush=True)
    print()
    print("Extracting …")
    with zipfile.ZipFile(buf) as z:
        z.extract(GLOVE_TXT)
    print(f"Extracted {GLOVE_TXT}")


def l2_norm(vec):
    mag = math.sqrt(sum(v * v for v in vec))
    return [v / mag for v in vec] if mag else vec


def main():
    if not os.path.exists(GLOVE_TXT):
        answer = input(f"{GLOVE_TXT} not found. Download automatically? [y/N] ").strip().lower()
        if answer == "y":
            download_glove()
        else:
            print(f"Place {GLOVE_TXT} in this directory and re-run.")
            sys.exit(1)

    words, flat = [], []
    in_top = set()

    print(f"Pass 1: reading top {MAX_WORDS} words …")
    with open(GLOVE_TXT, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= MAX_WORDS:
                break
            parts = line.split()
            if len(parts) != DIMS + 1:
                continue
            word = parts[0]
            vec  = l2_norm([float(x) for x in parts[1:]])
            words.append(word)
            flat.extend(round(v, 5) for v in vec)
            in_top.add(word)
            if (i + 1) % 2000 == 0:
                print(f"  {i + 1} …")

    missing_seeds = ALL_SEEDS - in_top
    if missing_seeds:
        print(f"Pass 2: scanning remainder for {len(missing_seeds)} seed words …")
        found_seeds = {}
        with open(GLOVE_TXT, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < MAX_WORDS:
                    continue
                parts = line.split()
                if len(parts) != DIMS + 1:
                    continue
                word = parts[0]
                if word in missing_seeds and word not in found_seeds:
                    found_seeds[word] = [float(x) for x in parts[1:]]
        for word, vec in sorted(found_seeds.items()):
            words.append(word)
            flat.extend(round(v, 5) for v in l2_norm(vec))
        print(f"  Added {len(found_seeds)} seed words from extended vocabulary.")

    print(f"Writing {OUTPUT_JS} …")
    with open(OUTPUT_JS, "w", encoding="utf-8") as f:
        f.write("// Auto-generated by generate-glove.py — do not edit\n")
        f.write(f"// GloVe 6B {DIMS}d · top {MAX_WORDS} words + curated Places/Food/People · L2-normalised\n")
        f.write("window.GLOVE_WORDS=")
        f.write(json.dumps(words, ensure_ascii=False))
        f.write(";\n")
        f.write(f"window.GLOVE_DIMS={DIMS};\n")
        f.write("window.GLOVE_VECS=new Float32Array(")
        f.write(json.dumps(flat))
        f.write(");\n")

    size = os.path.getsize(OUTPUT_JS) / 1024 / 1024
    print(f"Done — {OUTPUT_JS}  {size:.1f} MB  ({len(words)} words)")


if __name__ == "__main__":
    main()
