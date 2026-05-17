(function () {
    const FALLBACK = {
        options: {
            cities: ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"],
            area_types: ["Built Area", "Carpet Area", "Super Area"],
            furnishings: ["Furnished", "Semi-Furnished", "Unfurnished"],
            tenants: ["Bachelors", "Bachelors/Family", "Family"],
            contacts: ["Contact Agent", "Contact Builder", "Contact Owner"],
            bhk_min: 1, bhk_max: 6,
            size_min: 10, size_max: 8000,
            bath_min: 1, bath_max: 10
        },
        stats: {
            rows: 4746, cities: 6,
            avg_rent: 34917, median_rent: 16000,
            min_rent: 1200, max_rent: 3500000,
            city_avg: {
                "Kolkata": 11645, "Hyderabad": 20555, "Bangalore": 24966,
                "Chennai": 27508, "Delhi": 29867, "Mumbai": 85715
            }
        },
        submitted: {},
        prediction: null,
        error: null,
        model_info: {}
    };

    let DATA = FALLBACK;
    let isStaticPreview = true;
    try {
        const raw = document.getElementById('server-data').textContent.trim();
        if (raw && !raw.includes('{{') && raw.startsWith('{')) {
            const parsed = JSON.parse(raw);
            if (parsed && parsed.stats && parsed.options) {
                DATA = parsed;
                isStaticPreview = false;
            }
        }
    } catch (e) { /* fall through to fallback */ }

    if (isStaticPreview) {
        document.getElementById('preview-banner').style.display = 'block';
        document.getElementById('snapshot-source').textContent = 'example data';
    }

    const fmt = n => '₹' + Number(n).toLocaleString('en-IN');
    const opts = DATA.options;
    const stats = DATA.stats;
    const submitted = DATA.submitted || {};
    const modelInfo = DATA.model_info || {};

    document.getElementById('header-rows').textContent = Number(stats.rows).toLocaleString('en-IN');
    document.getElementById('header-cities').textContent = stats.cities;

    const bhk = document.getElementById('f-bhk');
    bhk.min = opts.bhk_min; bhk.max = opts.bhk_max;
    bhk.value = submitted.bhk || Math.max(opts.bhk_min, 2);

    const size = document.getElementById('f-size');
    size.min = opts.size_min; size.max = opts.size_max;
    size.value = submitted.size || Math.min(Math.max(opts.size_min, 1000), opts.size_max);

    const bath = document.getElementById('f-bathroom');
    bath.min = opts.bath_min; bath.max = opts.bath_max;
    bath.value = submitted.bathroom || Math.max(opts.bath_min, 2);

    function fillSelect(id, items, selected) {
        const el = document.getElementById(id);
        el.innerHTML = '';
        items.forEach(v => {
            const o = document.createElement('option');
            o.value = v; o.textContent = v;
            if (selected === v) o.selected = true;
            el.appendChild(o);
        });
    }
    fillSelect('f-city', opts.cities, submitted.city);
    fillSelect('f-area_type', opts.area_types, submitted.area_type);
    fillSelect('f-furnishing', opts.furnishings, submitted.furnishing);
    fillSelect('f-tenant', opts.tenants, submitted.tenant);
    fillSelect('f-contact', opts.contacts, submitted.contact);

    document.getElementById('s-rows').textContent = Number(stats.rows).toLocaleString('en-IN');
    document.getElementById('s-cities').textContent = stats.cities;
    document.getElementById('s-avg').textContent = fmt(stats.avg_rent);
    document.getElementById('s-median').textContent = fmt(stats.median_rent);
    document.getElementById('s-min').textContent = fmt(stats.min_rent);
    document.getElementById('s-max').textContent = fmt(stats.max_rent);

    const cityList = document.getElementById('city-list');
    Object.entries(stats.city_avg).forEach(([city, rent]) => {
        const li = document.createElement('li');
        const nameSpan = document.createElement('span');
        nameSpan.className = 'city-name';
        nameSpan.textContent = city;
        const rentSpan = document.createElement('span');
        rentSpan.className = 'city-rent';
        rentSpan.textContent = fmt(rent);
        li.appendChild(nameSpan);
        li.appendChild(rentSpan);
        cityList.appendChild(li);
    });

    if (modelInfo && modelInfo.test_r2 != null) {
        document.getElementById('model-info-card').style.display = 'block';
        document.getElementById('mi-r2').textContent = Number(modelInfo.test_r2).toFixed(3);
        document.getElementById('mi-mae').textContent = fmt(Math.round(modelInfo.test_mae));
        document.getElementById('mi-samples').textContent = Number(modelInfo.n_samples).toLocaleString('en-IN');
    }

    if (DATA.prediction) {
        document.getElementById('result-box').style.display = 'block';
        document.getElementById('result-value').textContent = DATA.prediction;
        if (modelInfo && modelInfo.test_mae != null) {
            document.getElementById('result-confidence').textContent =
                '±' + fmt(Math.round(modelInfo.test_mae)) + ' (model MAE)';
        }
    }
    if (DATA.error) {
        const errBox = document.getElementById('form-error');
        errBox.style.display = 'block';
        errBox.textContent = DATA.error;
    }

    const form = document.getElementById('rent-form');
    form.addEventListener('submit', function (e) {
        if (isStaticPreview) {
            e.preventDefault();
            alert('Static preview only. Run `python app.py` and open http://127.0.0.1:5000 to get live predictions.');
            return;
        }
        let ok = true;
        const checks = [
            ['f-bhk', opts.bhk_min, opts.bhk_max],
            ['f-size', opts.size_min, opts.size_max],
            ['f-bathroom', opts.bath_min, opts.bath_max],
        ];
        checks.forEach(([id, mn, mx]) => {
            const el = document.getElementById(id);
            const v = Number(el.value);
            const err = document.querySelector('.field-error[data-for="' + id + '"]');
            const bad = !el.value || isNaN(v) || v < mn || v > mx;
            el.classList.toggle('invalid', bad);
            if (err) {
                err.classList.toggle('show', bad);
                err.textContent = bad ? 'Enter a value between ' + mn + ' and ' + mx + '.' : '';
            }
            if (bad) ok = false;
        });
        if (!ok) e.preventDefault();
    });
})();
