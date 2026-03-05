const express = require('express');
const path = require('path');
const analyzeRoutes = require('./routes/analyze');

const app = express();

app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));
app.use(express.static(path.join(__dirname, 'public')));
app.use(express.urlencoded({ extended: true }));

app.get('/', (req, res) => res.render('index'));
app.get('/upload', (req, res) => res.render('upload'));
app.use('/analyze', analyzeRoutes);

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Frontend running on port ${PORT}`));