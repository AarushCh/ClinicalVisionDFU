const express = require('express');
const router = express.Router();
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');
const upload = multer();

const BACKEND_URL = process.env.BACKEND_URL || 'http://127.0.0.1:10000';

router.post('/', upload.single('image'), async (req, res) => {
    try {
        const form = new FormData();
        form.append('image', req.file.buffer, { filename: 'upload.jpg' });
        form.append('age', req.body.age);
        form.append('bmi', req.body.bmi);
        form.append('diabetes_years', req.body.diabetes_years);

        const response = await axios.post(`${BACKEND_URL}/predict`, form, {
            headers: { ...form.getHeaders() }
        });

        res.render('result', { data: response.data });
    } catch (error) {
        console.error(error);
        res.status(500).send("Error contacting AI Backend.");
    }
});

module.exports = router;