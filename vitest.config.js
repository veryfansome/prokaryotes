import { defineConfig } from 'vitest/config';

export default defineConfig({
    test: {
        environment: 'jsdom',
        include: ['tests/ui_tests/**/*.test.js'],
        globals: true,
        setupFiles: ['tests/ui_tests/setup.js'],
        coverage: {
            reporter: ['text', 'json-summary'],
        },
    },
});
