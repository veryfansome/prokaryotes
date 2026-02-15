import { defineConfig } from 'vitest/config';

export default defineConfig({
    test: {
        environment: 'jsdom',
        include: ['tests/**/*.test.js'],
        globals: true,
        setupFiles: ['tests/setup.js'],
        coverage: {
            reporter: ['text', 'json-summary'],
        },
    },
});
