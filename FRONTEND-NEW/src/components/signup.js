import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { toast } from 'react-toastify';
import styles from '../css/Login.module.css';
import API_BASE_URL from '../config';

const Signup = () => {
    const [formData, setFormData] = useState({
        organization: '',
        email: '',
        password: '',
    });
    const [error, setError] = useState('');
    const navigate = useNavigate();

    const handleChange = (event) => {
        const { name, value } = event.target;
        setFormData((prev) => ({
            ...prev,
            [name]: value,
        }));
    };

    const handleSubmit = async (event) => {
        event.preventDefault();
        setError('');

        try {
            const response = await fetch(`${API_BASE_URL}/signup`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData),
            });

            if (!response.ok) {
                const data = await response.json().catch(() => ({}));
                throw new Error(data.detail || 'Failed to create account.');
            }

            toast.success('Account created! You can log in now.');
            navigate('/login');
        } catch (err) {
            setError(err.message || 'Unexpected error. Please try again.');
            toast.error(err.message || 'Failed to create account. Please try again.');
        }
    };

    return (
        <div className={styles.container}>
            <form onSubmit={handleSubmit} className={styles.form}>
                <h2>Sign Up</h2>
                <input
                    type="text"
                    name="organization"
                    placeholder="Organization"
                    value={formData.organization}
                    onChange={handleChange}
                    required
                />
                <input
                    type="email"
                    name="email"
                    placeholder="Email"
                    value={formData.email}
                    onChange={handleChange}
                    required
                />
                <input
                    type="password"
                    name="password"
                    placeholder="Password"
                    minLength={8}
                    value={formData.password}
                    onChange={handleChange}
                    required
                />
                {error && <p className={styles.errorMessage}>{error}</p>}
                <button type="submit">Create Account</button>
                <p>
                    Already have an account?{' '}
                    <Link to="/login">Log in</Link>
                </p>
            </form>
        </div>
    );
};

export default Signup;
