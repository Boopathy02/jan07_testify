import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { toast } from 'react-toastify';
import styles from "../css/Login.module.css";
import API_BASE_URL from '../config';

const Login = () => {
    const [organization, setOrganization] = useState('');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const navigate = useNavigate();

    useEffect(() => {
        const token = localStorage.getItem('token');
        if (token) {
            navigate('/');
        }
    }, [navigate]);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        // Use dynamic API base URL (prop -> env -> default)
        try {
            const response = await fetch(`${API_BASE_URL}/login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    organization,
                    email,
                    password,
                }),
            });

            if (!response.ok) {
                const data = await response.json().catch(() => ({}));
                throw new Error(data.detail || 'Failed to login.');
            }

            const data = await response.json();
            localStorage.setItem('token', data.access_token);
            toast.success('Logged in successfully');
            navigate('/');
        } catch (err) {
            setError(err.message || 'Unexpected error. Please try again.');
            toast.error(err.message || 'Failed to login. Please try again.');
        }
    };

    return (
        <div className={styles.container}>
            <form onSubmit={handleSubmit} className={styles.form}>
                <h2>Login</h2>
                <input
                    type="text"
                    placeholder="Organization"
                    value={organization}
                    onChange={(e) => setOrganization(e.target.value)}
                    required
                />
                <input
                    type="email"
                    placeholder="Email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    required
                />
                <input
                    type="password"
                    placeholder="Password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required
                />
                {error && <p className={styles.errorMessage}>{error}</p>}
                <button type="submit">Login</button>
                <p>
                    Need an account?{' '}
                    <Link to="/signup">Sign up</Link>
                </p>
            </form>
        </div>
    );
};

export default Login;
